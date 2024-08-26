#!/usr/bin/env python
import json
from multiprocessing import process
from tkinter.tix import Tree
import jsonlines
import os
import cv2
import numpy as np
import yaml
import math
from visualization_msgs.msg import Marker, MarkerArray
import rospy
from lane_det.msg import lane_detection
from lane_det.msg import Localization
from geometry_msgs.msg import Point
from lane_det.cfg import LANE_PANELConfig
from dynamic_reconfigure.server import Server

class ImagePoint2Camera:
    """
    func  : 将图像坐标系下的点转换到相机坐标系下（假设:路面平）

    input : 内外参,误差范围
    return: 相机坐标系下的点
    """

    def __init__(self, int_param, ext_param, lane_eps_value, height):
        super(ImagePoint2Camera, self).__init__()
        self.int_param = int_param
        self.ext_param = ext_param
        self.lane_eps_value = lane_eps_value
        self.height = height

    def convert(self, point, delta_scale):
        point = np.array([point[0], point[1], 1.0])
        int_param_inverse = np.linalg.inv(self.int_param)
        org_camera_point = int_param_inverse @ point
        rotate_point = self.ext_param @ org_camera_point
        # apollo源代码中有异常值处理
        if abs(rotate_point[1]) < self.lane_eps_value:
            return None

        scale = self.height * 1.0 / rotate_point[1] * delta_scale

        # point in {camera}
        camera_point = np.array(
            [
                scale * org_camera_point[0],
                scale * org_camera_point[1],
                scale * org_camera_point[2],
            ]
        )

        return camera_point


class PubStaticLanes:
    def __init__(self) -> None:
        """
        load sensor config
        """
        # load internal and external param (from sensor.yaml)
        self.config_file = "/home/tsari/llx/Ultra-Fast-Lane-Detection/ros_ws/configs/sensors_amd64.yaml"
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.car_center_height = self.cfg["car_center_height"]
        # 车*道线的高度误差范围值
        self.lane_eps_value = self.cfg["lane_eps_value"]
        # 内参
        self.K = np.array(self.cfg["CameraMat"]["data"]).reshape(3, 3).T
        # 畸变参数
        self.D = np.array(self.cfg["DistCoeff"]["data"])
        # 外参 T_w_c
        self.R = np.array(self.cfg["CameraRotation"]["data"]).reshape(3, 3)
        self.t = np.array(self.cfg["CameraTranslation"]["data"]).reshape(3, 1)
        self.T_lidar_cam = np.vstack((np.hstack((self.R, self.t)), [0, 0, 0, 1]))

        self.T_car_lidar = np.vstack(
            (
                np.hstack(
                    (
                        np.array(self.cfg["Lidar"]["Rotation"]).reshape(3, 3),
                        np.array(self.cfg["Lidar"]["Translation"]).reshape(3, 1),
                    )
                ),
                [0, 0, 0, 1],
            )
        )

        self.T_car_cam = self.T_car_lidar @ self.T_lidar_cam
        self.T_lidar_cam0 = np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
        )

        self.T_cam0_cam = np.linalg.inv(self.T_lidar_cam0) @ self.T_lidar_cam

        self.image_topic = self.cfg["front_image_topic"]
        self.lidar_topic = self.cfg["lidar_topic"]
        self.lane_topic = self.cfg["lane_topic"]
        self.localization_topic = self.cfg["localization_topic"]

        # self.revise_theta = -1.1 * math.pi / 180
        # self.revise_y = -0.35
        # self.delta_scale = 1.0
        self.revise_lane = True
        
        # self.T_revise = np.array(
        #     [
        #         [math.cos(self.revise_theta), -1 * math.sin(self.revise_theta), 0, 0],
        #         [
        #             math.sin(self.revise_theta),
        #             math.cos(self.revise_theta),
        #             0,
        #             self.revise_y,
        #         ],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1],
        #     ]
        # )

        self.T_w_car = np.zeros(shape=(4, 4))
        
        # tag: rviz显示 原来的(天翼交通版本HDmap)偏移量是 x = 275000.02; y = 3479281.5
        #      最新的gtxc8.0版本的地图原点是 x = 275843.870151, y = 3477696.382524
        
        self.delta_x = 275843.870151
        self.delta_y = 3477696.382524
        
        self.lane_pub = rospy.Publisher(self.lane_topic, lane_detection, queue_size=10)
        # visualization lane in rviz
        self.lane_sub = rospy.Subscriber(
            self.lane_topic, lane_detection, self.rviz_visualization_callback
        )
        self.rviz_lane_left_1_pub = rospy.Publisher(
            "/rviz_lane_left_1", MarkerArray, queue_size=10
        )
        self.rviz_lane_left_0_pub = rospy.Publisher(
            "/rviz_lane_left_0", MarkerArray, queue_size=10
        )
        self.rviz_lane_right_0_pub = rospy.Publisher(
            "/rviz_lane_right_0", MarkerArray, queue_size=10
        )
        self.rviz_lane_right_1_pub = rospy.Publisher(
            "/rviz_lane_right_1", MarkerArray, queue_size=10
        )
        
        Server(LANE_PANELConfig, self.dynamic_callback)
    
    def dynamic_callback(self, config, level):
        """
        dynamic revise param
        """
        self.delta_scale = config["delta_scale"]
        self.revise_y = config["revise_y"]
        self.revise_theta = config["revise_theta"]
        
        self.T_revise = np.array(
            [
                [math.cos(self.revise_theta), -1 * math.sin(self.revise_theta), 0, 0],
                [
                    math.sin(self.revise_theta),
                    math.cos(self.revise_theta),
                    0,
                    self.revise_y,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
        return config
        

    def rviz_visualization_callback(self, lane_data: lane_detection):
        """
        visualize lane point by rviz marker
        """
        lane_left_1 = MarkerArray()
        lane_left_0 = MarkerArray()
        lane_right_0 = MarkerArray()
        lane_right_1 = MarkerArray()

        left_1 = []
        left_0 = []
        right_0 = []
        right_1 = []
        left_1 = lane_data.left_1
        left_0 = lane_data.left_0
        right_0 = lane_data.right_0
        right_1 = lane_data.right_1
        idx = 0

        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        for p in left_1:  # type: ignore
            marker.color.r = 1.0
            point = Point()
            point.x = p.x
            point.y = p.y
            point.z = p.z
            marker.points.append(point)  # type: ignore
        lane_left_1.markers.append(marker)  # type: ignore
        self.rviz_lane_left_1_pub.publish(lane_left_1)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for p in left_0:  # type: ignore
            marker.color.r = 1.0
            marker.color.g = 1.0
            point = Point()
            point.x = p.x
            point.y = p.y
            point.z = p.z
            marker.points.append(point)  # type: ignore
        lane_left_0.markers.append(marker)  # type: ignore
        self.rviz_lane_left_0_pub.publish(lane_left_0)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for p in right_0:  # type: ignore
            marker.color.b = 1.0
            point = Point()
            point.x = p.x
            point.y = p.y
            point.z = p.z
            marker.points.append(point)  # type: ignore
        lane_right_0.markers.append(marker)  # type: ignore
        self.rviz_lane_right_0_pub.publish(lane_right_0)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for p in right_1:  # type: ignore
            marker.color.g = 1.0
            marker.color.b = 1.0
            point = Point()
            point.x = p.x
            point.y = p.y
            point.z = p.z
            marker.points.append(point)  # type: ignore

        lane_right_1.markers.append(marker)  # type: ignore
        self.rviz_lane_right_1_pub.publish(lane_right_1)

    def quaternion_to_rotation_matrix(self, q):  # x, y ,z ,w
        """
        quaternion to rotation matrix
        """
        rot_matrix = np.array(
            [
                [
                    1.0 - 2 * (q[1] * q[1] + q[2] * q[2]),
                    2 * (q[0] * q[1] - q[3] * q[2]),
                    2 * (q[3] * q[1] + q[0] * q[2]),
                ],
                [
                    2 * (q[0] * q[1] + q[3] * q[2]),
                    1.0 - 2 * (q[0] * q[0] + q[2] * q[2]),
                    2 * (q[1] * q[2] - q[3] * q[0]),
                ],
                [
                    2 * (q[0] * q[2] - q[3] * q[1]),
                    2 * (q[1] * q[2] + q[3] * q[0]),
                    1.0 - 2 * (q[0] * q[0] + q[1] * q[1]),
                ],
            ]
        )
        return rot_matrix

    def process(self):
        T_cam0_cam = self.T_cam0_cam[:3, :3]
        image2camera = ImagePoint2Camera(
            self.K,
            T_cam0_cam,
            self.lane_eps_value,
            self.T_car_cam[2][3] + self.car_center_height,
        )
        
        # 用于话题发布
        lane_info = lane_detection()
        lane_left_1 = []
        lane_left_0 = []
        lane_right_1 = []
        lane_right_0 = []
        # 用于拟合多项式
        arr_lane_left_1_x = []
        arr_lane_left_0_x = []
        arr_lane_right_1_x = []
        arr_lane_right_0_x = []
        arr_lane_left_1_y = []
        arr_lane_left_0_y = []
        arr_lane_right_1_y = []
        arr_lane_right_0_y = []
                
        with open(
            "/home/tsari/llx/Ultra-Fast-Lane-Detection/ros_ws/data/images/img_loc.json",
            "r+",
        ) as f:
            for item in jsonlines.Reader(f):
                q = [item["qx"], item["qy"], item["qz"], item["qw"]]
                tmp_a = np.hstack(
                    (
                        self.quaternion_to_rotation_matrix(q),
                        [
                            [item["x"] - self.delta_x],
                            [item["y"] - self.delta_y],
                            [item["z"]],
                        ],
                    )
                )
                self.T_w_car = np.vstack((tmp_a, [0, 0, 0, 1]))
                
                

                
                for lane in item["lanes"]:
                    
                    if lane["label"] == "0":
                        for point in lane["points"]:
                            # point = lane["points"][0]
                            camera_point = image2camera.convert(point, self.delta_scale)

                            if camera_point is not None:
                                # point = [x, y, z, 1, class] ,i 补充作4*1向量，便于计算; class 代表车道线序号
                                camera_point = np.append(camera_point, 1.0)
                                car_point = self.T_car_cam @ camera_point
                                # revise results
                                if self.revise_lane:
                                    car_point = self.T_revise @ car_point

                                # convert point to {world}
                                car_point = self.T_w_car @ car_point

                                lane_point = Point()  # z = 0
                                lane_point.x = car_point[0]
                                lane_point.y = car_point[1]
                                
                                lane_left_1.append(lane_point)
                                arr_lane_left_1_x.append(lane_point.x)
                                arr_lane_left_1_y.append(lane_point.y)

                    if lane["label"] == "1":
                        # point = lane["points"][0]
                        for point in lane["points"]:
                            camera_point = image2camera.convert(point, self.delta_scale)

                            if camera_point is not None:
                                # point = [x, y, z, 1, class] ,i 补充作4*1向量，便于计算; class 代表车道线序号
                                camera_point = np.append(camera_point, 1.0)
                                car_point = self.T_car_cam @ camera_point
                                # revise results
                                if self.revise_lane:
                                    car_point = self.T_revise @ car_point

                                # convert point to {world}
                                car_point = self.T_w_car @ car_point

                                lane_point = Point()  # z = 0
                                lane_point.x = car_point[0]
                                lane_point.y = car_point[1]
                                
                                lane_left_0.append(lane_point)
                                arr_lane_left_0_x.append(lane_point.x)
                                arr_lane_left_0_y.append(lane_point.y)
                    
                    if lane["label"] == "2":
                        # point = lane["points"][0]
                        for point in lane["points"]:
                            camera_point = image2camera.convert(point, self.delta_scale)

                            if camera_point is not None:
                                # point = [x, y, z, 1, class] ,i 补充作4*1向量，便于计算; class 代表车道线序号
                                camera_point = np.append(camera_point, 1.0)
                                car_point = self.T_car_cam @ camera_point
                                # revise results
                                if self.revise_lane:
                                    car_point = self.T_revise @ car_point

                                # convert point to {world}
                                car_point = self.T_w_car @ car_point

                                lane_point = Point()  # z = 0
                                lane_point.x = car_point[0]
                                lane_point.y = car_point[1]
                                
                                lane_right_0.append(lane_point)
                                arr_lane_right_0_x.append(lane_point.x)
                                arr_lane_right_0_y.append(lane_point.y)
                    
                    if lane["label"] == "3":
                        # point = lane["points"][0]
                        for point in lane["points"]:
                            camera_point = image2camera.convert(point, self.delta_scale)

                            if camera_point is not None:
                                # point = [x, y, z, 1, class] ,i 补充作4*1向量，便于计算; class 代表车道线序号
                                camera_point = np.append(camera_point, 1.0)
                                car_point = self.T_car_cam @ camera_point
                                # revise results
                                if self.revise_lane:
                                    car_point = self.T_revise @ car_point

                                # convert point to {world}
                                car_point = self.T_w_car @ car_point

                                lane_point = Point()  # z = 0
                                lane_point.x = car_point[0]
                                lane_point.y = car_point[1]
                                
                                lane_right_1.append(lane_point)
                                arr_lane_right_1_x.append(lane_point.x)
                                arr_lane_right_1_y.append(lane_point.y)
                

            lane_info.left_0 = lane_left_0
            lane_info.left_1 = lane_left_1
            lane_info.right_0 = lane_right_0
            lane_info.right_1 = lane_right_1
                
            self.lane_pub.publish(lane_info)
            

def main():
    """
    main
    """
    rospy.init_node("test_lane_detection_node", anonymous=True)
    rospy.loginfo(" ***** test_lane_detection_node is running ***** ")
    pub_static_lanes = PubStaticLanes()
    while not rospy.is_shutdown():
        pub_static_lanes.process()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
