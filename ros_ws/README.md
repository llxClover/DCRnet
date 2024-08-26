### ssh连接远程服务器
vscode 连接 ssh： 192.168.1.55
```Shell
1. conda env list
2. conda activate lane-det

3. cd /home/tsari/llx/Ultra-Fast-Lane-Detection/ros_ws
```
### 服务器端
```Shell
1. roscore
# 新建终端
2. cd ~/llx/Ultra-Fast-Lane-Detection/ros_ws
3. source devel/setup.bash
4. rosrun lane_det ros_node.py
```
### 本地
```Shell
# 发布tf
1. rosrun tf static_transform_publisher 0.0 0.0 0.1 0.0 0.0 0.0 base_link map 100
# 新建终端
2. cd ~/Desktop
3. rviz -d lane_det.rviz
# 新建终端
4. cd /media/dell/0C247B76247B621E/llx/DATA/data_20231219/rosbags
5. rosbag play 2023-12-19-15-13-17.bag 
```
