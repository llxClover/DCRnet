"""
function: 测试自己的数据集，并保存成检测结果图
"""
import time
import csv
import argparse
import glob
from matplotlib import legend
import yaml
from tkinter import NO
import cv2
import os
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
import tqdm
from PIL import Image
import math
from data.constant import culane_row_anchor, tusimple_row_anchor
from data.dataset import LaneTestDataset
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import matplotlib.pyplot as plt

# 指定测试的配置信息
config_file = "./configs/sensors.yaml"
with open(config_file, "r", encoding="utf-8") as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

lane_num = cfg["lane_num"]
backbone = cfg["backbone"]
dataset = cfg["dataset"]
griding_num = cfg["griding_num"]
test_model = cfg["test_model"]
data_root = cfg["data_root"]
data_save = cfg["data_save"]
image_W, image_H = cfg["ImageSize"]
plot_path = cfg["plot_path"]
csv_path = cfg["csv_path"]
car_center_height = cfg["car_center_height"]
# 车道线的高度误差范围值
lane_eps_value = cfg["lane_eps_value"]

# 内参
K = np.array(cfg["CameraMat"]["data"]).reshape(3, 3).T

# 外参 T_w_c
R = np.array(cfg["CameraRotation"]["data"]).reshape(3, 3)
t = np.array(cfg["CameraTranslation"]["data"]).reshape(3, 1)

T_lidar_cam = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

T_car_lidar = np.vstack(
    (
        np.hstack(
            (
                np.array(cfg["Lidar"]["Rotation"]).reshape(3, 3),
                np.array(cfg["Lidar"]["Translation"]).reshape(3, 1),
            )
        ),
        [0, 0, 0, 1],
    )
)

T_car_cam = T_car_lidar @ T_lidar_cam

# print("-" * 100)
# print(T_car_lidar)
# print(T_lidar_cam)
# print(T_car_cam)
# print(T_car_cam[2][3])
T_lidar_cam0 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

T_cam0_cam = np.linalg.inv(T_lidar_cam0) @ T_lidar_cam


class TestDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, path, img_transform=None):
        super(TestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.img_list = glob.glob("%s/*.png" % self.path)

    def __getitem__(self, index):
        # glob模块的主要方法就是glob,该方法返回所有匹配的文件路径列表（list）
        name = glob.glob("%s/*.png" % self.path)[index]
        img = Image.open(name) #RGB

        if self.img_transform is not None:
            # print("TestDataset img size :  ", type(img),img.size) # (1920, 1080)

            img = self.img_transform(img)
        print("TestDataset img shape : ", img.shape) # type: ignore
        return img, name

    def __len__(self):
        return len(self.img_list)


class ImagePoint2Camera:
    """
    func  : 将图像坐标系下的点转换到相机坐标系下（假设:路面平）

    input : 内外参,误差范围
    return: 相机坐标系下的点
    """

    import numpy as np

    def __init__(self, int_param, ext_param, lane_eps_value, height):
        super(ImagePoint2Camera, self).__init__()
        self.int_param = int_param
        self.ext_param = ext_param
        self.lane_eps_value = lane_eps_value
        self.height = height

    def convert(self, point):
        point = np.array([point[0], point[1], 1.0])
        int_param_inverse = np.linalg.inv(self.int_param)
        org_camera_point = int_param_inverse @ point
        rotate_point = self.ext_param @ org_camera_point
        # apollo源代码中有异常值处理
        if abs(rotate_point[1]) < self.lane_eps_value:
            return None

        scale = self.height * 1.0 / rotate_point[1]

        # point in {camera}
        camera_point = np.array(
            [
                scale * org_camera_point[0],
                scale * org_camera_point[1],
                scale * org_camera_point[2],
            ]
        )

        return camera_point


def plot_3dlane(save_path, point_list, class_list):
    """
    func  : 绘制3D散点图

    input : array([[x0,y0,z0], [x1,y1,z1]])
    ouput : 3D 散点图
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
    from matplotlib.pyplot import MultipleLocator  # type: ignore
    import numpy as np

    x = point_list[:, 0]
    y = point_list[:, 1]
    z = point_list[:, 2]

    lane_num = [
        np.sum(class_list == 0),
        np.sum(class_list == 1),
        np.sum(class_list == 2),
        np.sum(class_list == 3),
    ]

    x_list = [
        x[: lane_num[0]],
        x[lane_num[0] : lane_num[0] + lane_num[1]],
        x[lane_num[0] + lane_num[1] : lane_num[0] + lane_num[1] + lane_num[2]],
        x[-lane_num[3] :],
    ]
    y_list = [
        y[: lane_num[0]],
        y[lane_num[0] : lane_num[0] + lane_num[1]],
        y[lane_num[0] + lane_num[1] : lane_num[0] + lane_num[1] + lane_num[2]],
        y[-lane_num[3] :],
    ]

    # 绘制BEV lane
    # plt.figure(figsize=(20, 10), dpi=80)
    plt.figure()
    # 定义车道线的颜色显示
    color = {
        0: "b",
        1: "g",
        2: "r",
        3: "c",
    }

    label = []
    for i in range(len(x_list)):
        s = plt.scatter(x_list[i], y_list[i], c=color[i])
        label.append(s)
    plt.plot(0, 0, marker=">", color="r", markersize=15)
    plt.text(-0.4, 0.3, "Car", size=12)
    plt.arrow(3, 0, 2, 0, width=0.2, color="b")
    plt.text(3.0, 0.2, "Front", size=12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BEV")

    plt.legend(tuple(label), ("lane_0", "lane_1", "lane_2", "lane_3"), loc="right")
    # 设置坐标轴格式
    x_major_locator = MultipleLocator(1.0)
    y_major_locator = MultipleLocator(1.0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_aspect(1)
    # plt.xlim(-1,50)
    # plt.ylim(-5,5)
    plt.grid()

    plt.savefig(save_path)
    plt.close()

    # plt.show()

def save_csv(path, data):
    with open(path,"w") as csvfile: 
        writer = csv.writer(csvfile)
        for d in data:
            writer.writerow(d)
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # type: ignore # 加速
    # args, cfg = merge_config()   # 用终端指定配置信息
    dist_print("start testing...")
    assert backbone in [
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

    if dataset == "CULane":
        cls_num_per_lane = 18
    elif dataset == "Tusimple":
        cls_num_per_lane = 56
    else:
        # raise NotImplementedError
        cls_num_per_lane = 56
    # TODO： 测试时，输出seg图，进行变道校正use_aux=True
    net = parsingNet(
        pretrained=False,
        backbone=backbone,
        cls_dim=(griding_num + 1, cls_num_per_lane, lane_num),
        use_aux=True,
    ).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(test_model, map_location="cpu")["model"]
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    # 图像格式统一：(288, 800)，图像张量，归一化
    img_transforms = transforms.Compose(
        [
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if dataset == "CULane":
        splits = [
            "test0_normal.txt",
            "test1_crowd.txt",
            "test2_hlight.txt",
            "test3_shadow.txt",
            "test4_noline.txt",
            "test5_arrow.txt",
            "test6_curve.txt",
            "test7_cross.txt",
            "test8_night.txt",
        ]
        datasets = [
            LaneTestDataset(
                data_root,
                os.path.join(data_root, "list/test_split/" + split),
                img_transform=img_transforms,
            )
            for split in splits
        ]

        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif dataset == "Tusimple":
        splits = ["test.txt"]
        datasets = [
            LaneTestDataset(
                data_root, os.path.join(data_root, split), img_transform=img_transforms
            )
            for split in splits
        ]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:  # 自定义数据集
        # raise NotImplementedError
        datasets = TestDataset(data_root, img_transform=img_transforms)
        # 图像的分辨率大小
        img_w, img_h = image_W, image_H
        row_anchor = tusimple_row_anchor
    print("-"*100)
    for dataset in zip(datasets):  # splits：图片列表 datasets：统一格式之后的数据集
        print("*"*10)
        # print(dataset)
        loader = torch.utils.data.DataLoader(  # type: ignore
            dataset, batch_size=1, shuffle=False, num_workers=1
        )  # 加载数据集
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # print(split[:-3]+'avi')
        # vout = cv2.VideoWriter(split[:-3]+'avi', fourcc, 30.0, (img_w, img_h))  # 保存结果为视频文件
        for i, data in enumerate(tqdm.tqdm(loader)):  # 进度条显示进度
            imgs, names = data  # imgs:图像张量，图像相对路径：
            print("==> ", imgs.shape, type(imgs)) #  torch.Size([1, 3, 288, 800])
            imgs = imgs.cuda()  # 使用GPU
            print("-"*50)
            print(imgs.shape) # torch.Size([1, 3, 288, 800])
            
            start_time = time.time()
            with torch.no_grad():  # 测试代码不计算梯度
                # TODO: 输出seg掩码
                out, seg= net(imgs)  # 模型预测 输出张量：[1,101,56,4]，掩码
                print("seg.shape : ", seg.shape)
            
            col_sample = np.linspace(0, 800 - 1, griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            out_j = out[0].data.cpu().numpy()  # 数据类型转换成numpy [101,56,4]
            out_j = out_j[:, ::-1, :]  # 将第二维度倒着取[101,56,4]
            prob = scipy.special.softmax(
                out_j[:-1, :, :], axis=0
            )  # [100,56,4]softmax 计算（概率映射到0-1之间且沿着维度0概率总和=1）
            idx = np.arange(griding_num) + 1  # 产生 1-100
            idx = idx.reshape(-1, 1, 1)  # [100,1,1]
            loc = np.sum(prob * idx, axis=0)  # [56,4]
            out_j = np.argmax(out_j, axis=0)  # 返回最大值的索引
            loc[out_j == griding_num] = 0  # 若最大值的索引=griding_num，归零
            out_j = loc  # [56,4]
            end_time = time.time()
            print("*"*50)
            print("inference frequency :  " , 1/ (end_time - start_time))
            print("*"*50)
            # import pdb; pdb.set_trace()
            T_cam0_cam = T_cam0_cam[:3, :3]
            image2camera = ImagePoint2Camera(
                K, T_cam0_cam, lane_eps_value, 
                T_car_cam[2][3] + car_center_height
            )

            # point = [x, y, z, 1, class] ,i 补充作4*1向量，便于计算; class 代表车道线序号
            car_points = np.empty([0, 5])
            vis = cv2.imread(os.path.join(data_root, names[0]))  # 读取图像
            # 定义车道线的颜色显示
            color = {
                0: (255, 0, 0),
                1: (0, 255, 0),
                2: (0, 0, 255),
                3: (255, 255, 0),
            }

            for i in range(out_j.shape[1]):  # 遍历列# C 车道线数
                if np.sum(out_j[:, i] != 0) > 2:  # 非0单元格的数量大于2
                    sum1 = np.sum(out_j[:, i] != 0)
                    for k in range(out_j.shape[0]):  # 遍历行row_anchor:56
                        if out_j[k, i] > 0:
                            point = (
                                int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                                int(row_anchor[cls_num_per_lane - 1 - k] * img_h / 288)
                                - 1,
                            )
                            # TODO: point 就是需要2D识别的车道线点
                            camera_point = image2camera.convert(point)

                            if camera_point is not None:
                                # point = [x, y, z, 1, class] ,i 补充作4*1向量，便于计算; class 代表车道线序号
                                camera_point = np.append(camera_point, 1.0)
                                car_point = T_car_cam @ camera_point
                                car_points = np.vstack(
                                    (car_points, np.append(car_point, i))
                                )  # 加上类别

                            cv2.circle(vis, point, 5, color[i], -1)
            # 保存检测结果图
            # print(os.path.join(data_save, os.path.basename(names[0])))
            cv2.imwrite(os.path.join(data_save, os.path.basename(names[0])), vis)

            # 绘制3D 车道线散点图
            points = car_points[:, :3]
            point_class = car_points[:, -1]
            save_csv(csv_path + "/" + names[0][-21:]+".csv", car_points)
            plot_3dlane(plot_path + "/" + names[0][-21:], points, point_class)
        # 保存视频结果（注释掉）
        #     vout.write(vis)
        #
        # vout.release()
