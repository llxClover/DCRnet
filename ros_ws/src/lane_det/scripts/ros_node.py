#!/usr/bin/env python3
from email.mime import image
import json
from re import I, L
import time
from tkinter import image_names
from xml.dom import INDEX_SIZE_ERR
import rospkg
from visualization_msgs.msg import Marker, MarkerArray
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import yaml
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from lane_det.msg import lane_detection
from lane_det.msg import Localization
from geometry_msgs.msg import Point
import scipy.special
import sys
import math
import transforms3d as tfs
import message_filters
from lane_det.cfg import LANE_PANELConfig
from dynamic_reconfigure.server import Server

sys.path.append("/home/tsari/llx/Ultra-Fast-Lane-Detection")
from model.model import parsingNet


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


class LaneDetection:
    """
    ros node of lane detection
    """

    def __init__(self) -> None:
        """
        load sensor config
        """
        # load internal and external param (from sensor.yaml)
        self.config_file = "/home/tsari/llx/Ultra-Fast-Lane-Detection/ros_ws/configs/sensors_amd64.yaml"
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.row_anchor = self.cfg["tusimple_row_anchor"]
        self.lane_num = self.cfg["lane_num"]
        self.backbone = self.cfg["backbone"]
        self.dataset = self.cfg["dataset"]
        self.griding_num = self.cfg["griding_num"]
        self.test_model = self.cfg["test_model"]
        self.data_root = self.cfg["data_root"]
        self.data_save = self.cfg["data_save"]
        self.image_W, self.image_H = self.cfg["ImageSize"]
        self.plot_path = self.cfg["plot_path"]
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

        # self.revise_theat = -0.03
        # self.revise_y = -0.2
        self.revise_lane = True
        # self.T_revise = np.array(
        #     [
        #         [math.cos(self.revise_theat), -1 * math.sin(self.revise_theat), 0, 0],
        #         [
        #             math.sin(self.revise_theat),
        #             math.cos(self.revise_theat),
        #             0,
        #             self.revise_y,
        #         ],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1],
        #     ]
        # )

        torch.backends.cudnn.benchmark = True  # type: ignore # 加速
        assert self.backbone in [
            "18",
            "34",
            "50",
            "101",
            "152",
            "50next",
            "101next",
            "50wide",
            "101wide",
        ]

        self.cls_num_per_lane = 56
        self.net = parsingNet(
            pretrained=False,
            backbone=self.backbone,
            cls_dim=(self.griding_num + 1, self.cls_num_per_lane, self.lane_num),
            use_aux=False,
        ).cuda()  # we dont need auxiliary segmentation in testing

        state_dict = torch.load(self.test_model, map_location="cpu")["model"]
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()
        # 图像格式统一：(288, 800)，图像张量，归一化
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((288, 800)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # self.row_anchor = tusimple_row_anchor
        print("=" * 50)
        print(" " * 20 + "net is ready")
        print("=" * 50)

        # self.image_sub = rospy.Subscriber(
        #     self.image_topic, CompressedImage, self.image_callback
        # )
        # self.localization_sub = rospy.Subscriber(
        #     self.localization_topic, Localization, self.localization_callback
        # )

        self.sub_img = message_filters.Subscriber(self.image_topic, CompressedImage)
        self.sub_loc = message_filters.Subscriber(self.localization_topic, Localization)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_loc], 10, 1
        )
        self.sync.registerCallback(self.img_loc_callback)

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
        self.rviz_car_pub = rospy.Publisher("/car", Marker, queue_size=10)

        self.car_position = Localization()
        self.T_w_car = np.zeros(shape=(4, 4))

        self.delta_x = 275843.870151
        self.delta_y = 3477696.382524
        self.pow_num = 1
        
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
        
        
    def elimiante_distortion(self, image):
        """
        eliminate raw image distortion
        """

        h, w = image.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (w, h), 5)  # type: ignore
        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    def polynomial(self, abcd, x):
        """
        input : coeff of polynomial and x,
        output : y
        """
        if self.pow_num == 1:
            return abcd[0] * x + abcd[1] 
        elif self.pow_num == 2:
            return abcd[0] * pow(x, 2) + abcd[1] * x + abcd[2]
        else:
            return abcd[0] * pow(x, 3) + abcd[1] * pow(x, 2) + abcd[2] * x + abcd[3]
    
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

    def backward_extension(self, abcd, min_x, delta_x):
        """
        input: coeff of cubic polynomial, end point x, delta_x
        ouput: result arrary of backward extension
        """
        points = []
        extension_length = 7
        for i in range(int(extension_length / delta_x), 0, -1):
            p = Point()
            p.x = min_x - i * delta_x
            p.y = self.polynomial(abcd, p.x)
            points.append(p)

        return points

    def inference(self, image):
        """
        inference image by trained moudle.
        """
        print("\n")
        print("-" * 50)
        rospy.loginfo("   *****  inferenceing ...  *****")

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # type: ignore
        # 在训练时的数据维度一般都是 (batch_size, c, h, w)，
        # 而在测试时只输入一张图片(c,h,w)，所以需要扩展维度
        # add batch size channel

        # # 图像格式统一：(288, 800)，图像张量，归一化
        # img_transforms = transforms.Compose(
        #     [
        #         transforms.Resize((288, 800)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     ]
        # )
        imgs = self.img_transforms(image).unsqueeze(dim=0).cuda()

        with torch.no_grad():  # 测试代码不计算梯度
            out = self.net(imgs)  # 模型预测 输出张量：[1,101,56,4]

        col_sample = np.linspace(0, 800 - 1, self.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()  # 数据类型转换成numpy [101,56,4]
        out_j = out_j[:, ::-1, :]  # 将第二维度倒着取[101,56,4]
        prob = scipy.special.softmax(
            out_j[:-1, :, :], axis=0
        )  # [100,56,4]softmax 计算（概率映射到0-1之间且沿着维度0概率总和=1）
        idx = np.arange(self.griding_num) + 1  # 产生 1-100
        idx = idx.reshape(-1, 1, 1)  # [100,1,1]
        loc = np.sum(prob * idx, axis=0)  # [56,4]
        out_j = np.argmax(out_j, axis=0)  # 返回最大值的索引
        loc[out_j == self.griding_num] = 0  # 若最大值的索引=griding_num，归零
        out_j = loc  # [56,4]

        T_cam0_cam = self.T_cam0_cam[:3, :3]
        image2camera = ImagePoint2Camera(
            self.K,
            T_cam0_cam,
            self.lane_eps_value,
            self.T_car_cam[2][3] + self.car_center_height,
        )

        # point = [x, y, z, 1, class] ,i 补充作4*1向量，便于计算; class 代表车道线序号
        car_points = np.empty([0, 5])
        lane_info = lane_detection()
        # 用于话题发布
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

        for i in range(out_j.shape[1]):  # 遍历列# C 车道线数
            if np.sum(out_j[:, i] != 0) > 2:  # 非0单元格的数量大于2
                sum1 = np.sum(out_j[:, i] != 0)
                for k in range(out_j.shape[0]):  # 遍历行row_anchor:56
                    if out_j[k, i] > 0:
                        point = (
                            int(out_j[k, i] * col_sample_w * self.image_W / 800) - 1,
                            int(
                                self.row_anchor[self.cls_num_per_lane - 1 - k]
                                * self.image_H
                                / 288
                            )
                            - 1,
                        )
                        # TAG: point 就是需要2D识别的车道线点

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
                            if i == 0:
                                lane_left_1.append(lane_point)
                                arr_lane_left_1_x.append(lane_point.x)
                                arr_lane_left_1_y.append(lane_point.y)
                            elif i == 1:
                                lane_left_0.append(lane_point)
                                arr_lane_left_0_x.append(lane_point.x)
                                arr_lane_left_0_y.append(lane_point.y)
                            elif i == 2:
                                lane_right_0.append(lane_point)
                                arr_lane_right_0_x.append(lane_point.x)
                                arr_lane_right_0_y.append(lane_point.y)
                            elif i == 3:
                                lane_right_1.append(lane_point)
                                arr_lane_right_1_x.append(lane_point.x)
                                arr_lane_right_1_y.append(lane_point.y)

                            car_points = np.vstack(
                                (car_points, np.append(car_point, i))
                            )  # 加上类别

        # 拟合多项式
        if len(arr_lane_left_0_x) != 0 and not all(x == 0 for x in arr_lane_left_0_x):
            lane_info.coeff_left_0 = np.polyfit(arr_lane_left_0_x, arr_lane_left_0_y,self.pow_num)

            min_x = min(arr_lane_left_0_x)
            res = self.backward_extension(lane_info.coeff_left_0, min_x, 0.2)
            lane_left_0 = res + lane_left_0

        lane_info.left_0 = lane_left_0

        if len(arr_lane_left_1_x) != 0 and arr_lane_left_1_x[0] != 0:
            lane_info.coeff_left_1 = np.polyfit(arr_lane_left_1_x, arr_lane_left_1_y,self.pow_num)
            min_x = min(arr_lane_left_1_x)
            res = self.backward_extension(lane_info.coeff_left_1, min_x, 0.2)
            lane_left_1 = res + lane_left_1
        lane_info.left_1 = lane_left_1

        if len(arr_lane_right_0_x) != 0 and arr_lane_right_0_x[0] != 0:
            lane_info.coeff_right_0 = np.polyfit(arr_lane_right_0_x, arr_lane_right_0_y,self.pow_num)
            min_x = min(arr_lane_right_0_x)
            res = self.backward_extension(lane_info.coeff_right_0, min_x, 0.2)
            lane_right_0 = res + lane_right_0
        lane_info.right_0 = lane_right_0

        if len(arr_lane_right_1_x) != 0 and arr_lane_right_1_x[0] != 0:
            lane_info.coeff_right_1 = np.polyfit(arr_lane_right_1_x, arr_lane_right_1_y,self.pow_num)
            min_x = min(arr_lane_right_1_x)
            res = self.backward_extension(lane_info.coeff_right_1, min_x, 0.2)
            lane_right_1 = res + lane_right_1
        lane_info.right_1 = lane_right_1

        self.lane_pub.publish(lane_info)
        # print(car_points)
        return car_points

    def save_img_with_loc(self, img:CompressedImage, loc: Localization):
        """
        save image with location & timestamp
        """
        image_path = "/home/tsari/llx/Ultra-Fast-Lane-Detection/ros_ws/data/images/"
        image = CvBridge().compressed_imgmsg_to_cv2(img)
        image = self.elimiante_distortion(image)
        image_name = str("%.10f" % img.header.stamp.to_sec())
        
        cv2.imwrite(image_path + image_name + '.png', image)
        
        loc = {
            "x": loc.position.x,
            "y": loc.position.y,
            "z": loc.position.z,
            "w": loc.orientation.w,
            "qx": loc.orientation.x,
            "qy": loc.orientation.y,
            "qz": loc.orientation.z,
            "loc_timestamp": "%.10f" % loc.header.stamp.to_sec(),
            "img_timestamp": "%.10f" % img.header.stamp.to_sec(),
        }  # type: ignore
        
        loc_json = json.dumps(loc)
        f = open(image_path + 'location.json', 'a+')
        f.write(loc_json + "\n")
        f.close()
        
        

    def img_loc_callback(self, image_data:CompressedImage, localization_data: Localization):
        """
        sync callback img & loc
        """
        raw_image = CvBridge().compressed_imgmsg_to_cv2(image_data)
        time_0 = time.time()
        image = self.elimiante_distortion(raw_image)
        time_1 = time.time()
        print("elimiante distortion cost time :==> ", time_1 - time_0)
        results = self.inference(image)
        time_2 = time.time()
        print("inference cost time :==> ", time_2 - time_1)

        self.car_position.position = localization_data.position
        self.car_position.orientation = localization_data.orientation

        q = []
        q.append(self.car_position.orientation.x)
        q.append(self.car_position.orientation.y)
        q.append(self.car_position.orientation.z)
        q.append(self.car_position.orientation.w)

        # q = [0,0,0,1]

        tmp_a = np.hstack(
            (
                self.quaternion_to_rotation_matrix(q),
                [
                    [localization_data.position.x - self.delta_x],
                    [localization_data.position.y - self.delta_y],
                    [localization_data.position.z],
                ],
            )
        )

        self.T_w_car = np.vstack((tmp_a, [0, 0, 0, 1]))
        # print("T_w_car  = \n", self.T_w_car)

        self.save_img_with_loc(image_data, localization_data)
        
    def image_callback(self, image_data):
        """
        image topic callback
        """
        raw_image = CvBridge().compressed_imgmsg_to_cv2(image_data)
        time_0 = time.time()
        image = self.elimiante_distortion(raw_image)
        time_1 = time.time()
        print("elimiante distortion cost time :==> ", time_1 - time_0)
        results = self.inference(image)
        time_2 = time.time()
        print("inference cost time :==> ", time_2 - time_1)

    def localization_callback(self, localization_data: Localization):
        """
        localization topic callback
        """
        self.car_position.position = localization_data.position
        self.car_position.orientation = localization_data.orientation

        q = []
        q.append(self.car_position.orientation.x)
        q.append(self.car_position.orientation.y)
        q.append(self.car_position.orientation.z)
        q.append(self.car_position.orientation.w)

        tmp_a = np.hstack(
            (
                self.quaternion_to_rotation_matrix(q),
                [
                    [localization_data.position.x - self.delta_x],
                    [localization_data.position.y - self.delta_y],
                    [localization_data.position.z],
                ],
            )
        )

        self.T_w_car = np.vstack((tmp_a, [0, 0, 0, 1]))

    def rviz_visualization_callback(self, lane_data: lane_detection):
        """
        visualize lane point by rviz marker
        """

        car = Marker()
        car.header.frame_id = "map"
        car.type = Marker.CUBE
        car.action = car.ADD
        car.id = 500
        car.ns = "car"
        car.scale.x = 4.854
        car.scale.y = 1.995
        car.scale.z = 1.703
        car.color.a = 1.0
        car.color.r = 1.0
        car.color.g = 1.0
        car.color.b = 1.0
        car.pose.orientation.w = 1.0
        car.pose.position.x = self.car_position.position.x - self.delta_x
        car.pose.position.y = self.car_position.position.y - self.delta_y
        car.pose.position.z = 0

        self.rviz_car_pub.publish(car)

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
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.1
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
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.1
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
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.1
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
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.id = idx
        idx += 1
        marker.ns = "lane"
        marker.scale.x = 0.1
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


def main():
    """
    main
    """
    rospy.init_node("lane_detection_node", anonymous=True)
    lane_det = LaneDetection()
    rospy.loginfo(" ***** lane_detection_node is running ***** ")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
