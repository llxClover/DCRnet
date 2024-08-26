
------
 # 使用
 ```Shell
 cd ~/llx/Ultra-Fast-Lane-Detection/
 source ~/.bashrc
 conda env list
 conda activate lane-det
 ```
 ## 训练
 ```Shell
  python train.py configs/tusimple.py --data_root '/home/tsari/llx/Ultra-Fast-Lane-Detection/data/Tusimple/TUSimple/train_set'
 ```
 ## 打开tensorboard
 ```Shell
 tensorboard --logdir /media/tsari/HIKVISION/llx/log --host localhost
 ```
 ## 测试
 ```Shell
 python test.py configs/tusimple.py --data_root '/home/tsari/llx/Ultra-Fast-Lane-Detection/data/Tusimple/TUSimple/test_set' --test_model ep099.pth --test_work_dir ./tmp 
 ```
## 可视化
```Shell
python demo.py configs/tusimple.py --test_model ep099.pth
```
<br/>
<br/>

## 测试自己的图像数据
```Shell
# 注意： 内部参数的配置文件在configs/sensors.yaml
python demo_custum.py 

```
 
