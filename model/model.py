import torch
from model.backbone import resnet
import numpy as np
from registry import build_backbones, build_aggregator, build_heads, build_necks

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False,use_fpn=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        if use_fpn:
            self.fpn = build_necks()

    def forward(self,x):
        x = self.conv(x)
        # print("conv => :", x.shape)
        x = self.bn(x)
        # print("bn => :", x.shape)
        x = self.relu(x)
        # print("relu => :", x.shape)
        return x
class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        # input : nchw,
        # output: (w+1) * sample_rows * 4 
        self.model = resnet(backbone, pretrained=pretrained)

        if self.use_aux:
            # self.aux_header2 = torch.nn.Sequential(
            #     conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            #     conv_bn_relu(128,128,3,padding=1),
            #     conv_bn_relu(128,128,3,padding=1),
            #     conv_bn_relu(128,128,3,padding=1),
            # )
            # self.aux_header3 = torch.nn.Sequential(
            #     conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            #     conv_bn_relu(128,128,3,padding=1),
            #     conv_bn_relu(128,128,3,padding=1),
            # )
            # self.aux_header4 = torch.nn.Sequential(
            #     conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            #     conv_bn_relu(128,128,3,padding=1),
            # )
            # self.aux_combine = torch.nn.Sequential(
            #     conv_bn_relu(384, 256, 3,padding=2,dilation=2),
            #     conv_bn_relu(256, 128, 3,padding=2,dilation=2),
            #     conv_bn_relu(128, 128, 3,padding=2,dilation=2),
            #     conv_bn_relu(128, 128, 3,padding=4,dilation=4),
            #     torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
            #     # output : n, num_of_lanes+1, h, w
            # )
            
            # initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)
            
            
            self.aux_header0 = torch.nn.Sequential(
                conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(32,32,3,padding=1),
                conv_bn_relu(32,32,3,padding=1)
            )          
            self.aux_header1 = torch.nn.Sequential(
                conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(32,32,3,padding=1),
                conv_bn_relu(32,32,3,padding=1)
            ) 
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 64, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(64,64,3,padding=1),
                conv_bn_relu(64,64,3,padding=1)
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1)
            )
            
            self.aux_1x1_4 = torch.nn.Sequential(
                conv_bn_relu(640, 256, kernel_size=1, stride=1, padding=0),
                conv_bn_relu(256,256,1,padding=0),
                conv_bn_relu(256,256,1,padding=0)
            )
            self.aux_1x1_3 = torch.nn.Sequential(
                conv_bn_relu(320, 128, kernel_size=1, stride=1, padding=0),
                conv_bn_relu(128,128,1,padding=0),
                conv_bn_relu(128,128,1,padding=0)
            ) 
            
            self.aux_1x1_2 = torch.nn.Sequential(
                conv_bn_relu(192, 64, kernel_size=1, stride=1, padding=0),
                conv_bn_relu(64,64,1,padding=0),
                conv_bn_relu(64,64,1,padding=0)
            )
            self.aux_1x1_1 = torch.nn.Sequential(
                conv_bn_relu(64, 32, kernel_size=1, stride=1, padding=0),
                conv_bn_relu(32,32,1,padding=0),
                conv_bn_relu(32,32,1,padding=0),
                torch.nn.Conv2d(32, cls_dim[-1] + 1,1)
            )

            initialize_weights(self.aux_header0,self.aux_header1,self.aux_header2,self.aux_header3,
                               self.aux_1x1_4,self.aux_1x1_3,self.aux_1x1_2,self.aux_1x1_1)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)

    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x0,x1,x2,x3,fea = self.model(x)
        # print("x.shape : ", x.shape) #[32, 3, 288, 800]
        # print("x1,x2,x3,fea.shape :", x1.shape, x2.shape, x3.shape, fea.shape)
        if self.use_aux:
            # x2 = self.aux_header2(x2)
            # print("x2.shape : ", x2.shape) #[32, 128, 36, 100]
            # x3 = self.aux_header3(x3)
            # print("x3.shape : ", x3.shape) #[32, 128, 18, 50]
            # x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            # print("x3_1.shape : ", x3.shape) #[32, 128, 36, 100]
            # x4 = self.aux_header4(fea)
            # print("x4.shape : ", x4.shape)#[32, 128, 9, 25]
            # x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            # print("x4_1.shape : ", x4.shape) #[32, 128, 36, 100]
            # aux_seg = torch.cat([x2,x3,x4],dim=1)
            # print("aux_seg.shape : ", aux_seg.shape) #[32, 384, 36, 100]
            # aux_seg = self.aux_combine(aux_seg)
            # print("aux_seg_1.shape : ", aux_seg.shape) #[32, 5, 36, 100]
            
            x_0 = self.aux_header0(x0)
            # print("x_0.shape : ", x_0.shape)
            x_1 = self.aux_header1(x1)
            # print("x_1.shape : ", x_1.shape)
            x_2 = self.aux_header2(x2)
            # print("x_2.shape : ", x_2.shape)
            x_3 = self.aux_header3(x3)
            # print("x_3.shape : ", x_3.shape)
            x_4 = fea
            # print("x_4.shape : ", x_4.shape)
            
            x4_2 = torch.nn.functional.interpolate(x_4,scale_factor = 2,mode='bilinear')
            # print("x4_2.shape : ", x4_2.shape)
            x4_2_cat_x3 = torch.cat([x4_2,x_3],dim=1)
            # print("x4_2_cat_x3.shape : ", x4_2_cat_x3.shape)
            x4_2_cat_x3 = self.aux_1x1_4(x4_2_cat_x3)
            # print("x4_2_cat_x3.shape : ", x4_2_cat_x3.shape)
            
            x3_2 = torch.nn.functional.interpolate(x4_2_cat_x3,scale_factor = 2,mode='bilinear')
            # print("x3_2.shape : ", x3_2.shape)
            x3_2_cat_x2 = torch.cat([x3_2,x_2],dim=1)
            # print("x3_2_cat_x2.shape : ", x3_2_cat_x2.shape)
            x3_2_cat_x2 = self.aux_1x1_3(x3_2_cat_x2)   
            # print("x3_2_cat_x2.shape : ", x3_2_cat_x2.shape)         

            x2_2 = torch.nn.functional.interpolate(x3_2_cat_x2,scale_factor = 2,mode='bilinear')
            # print("x2_2.shape : ", x2_2.shape) 
            x2_2_cat_x1_x0 = torch.cat([x2_2,x_1,x_0],dim=1)
            # print("x2_2_cat_x1_x0.shape : ", x2_2_cat_x1_x0.shape) 
            x2_2_cat_x1_x0 = self.aux_1x1_2(x2_2_cat_x1_x0) 
            # print("x2_2_cat_x1_x0.shape : ", x2_2_cat_x1_x0.shape) 
            
            x1_2 = torch.nn.functional.interpolate(x2_2_cat_x1_x0,scale_factor = 4,mode='bilinear')
            # print("x1_2.shape : ", x1_2.shape) 
            aux_seg = self.aux_1x1_1(x1_2) 
            # print("aux_seg.shape : ", aux_seg.shape)            
        else:
            aux_seg = None
        

        fea = self.pool(fea).view(-1, 1800)
        
        if use_fpn:
            fea = self.fpn(fea)
            
        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
