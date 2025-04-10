'''
该文件存放所有的全局变量数据
'''
from numbers import Rational
from xmlrpc.client import TRANSPORT_ERROR
import numpy as np

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
|                       重要参数调整区域                            |
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 敌我双方颜色，手动更改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 2023_08_11 12_09 08_12 08_30
# 2023_08_10 12_13
# 2023_08_09 11_16
# 2023_06_08 17_14
# 2023_06_09 12_00
# 2023_06_10 09_31
# 2023_06_10 14_23
# 2023_06_11 12_41
# 2023_06_11 19_24
# 2023_06_12 10_29
# 2023_06_12 13_23
reverse: int = 1 # 0: 我方 R, 敌方 B
                 # 1: 我方 B, 敌方 R

using_serial: bool = False # 裁判系统串口通信
test: bool = True # 使用录制的视频进行测试
BO: int = 3 # Max 比赛局数, 小组赛 2, 淘汰赛 3, 决赛 5
demo: bool = True # 主画面bbx与Type显示
debug_info: bool = False
open_track: bool = False

##### advanced 测试参 #####
# 5.5 三个相机换不同的视频源
if test:
    home_test = False  # 是否是在家里测试(用缩小版尺寸)
    using_video = True  # 测试是否用视频  
    # VIDEO_PATH = "demo_resource/videos/RMUC_2022_0627_04.mp4" # RED
    VIDEO_PATH = ["demo_resource/videos/0518_0.mp4", "demo_resource/videos/0518_2.avi"] # BLUE
    # VIDEO_PATH = "demo_resource/videos/sjtu.mp4"
    using_dq = True # 使用record的点云来代替实际的radar节点输入
    # PC_RECORD_SAVE_DIR = "demo_resource/RMUC_2022_0627_04.pkl" # RED
    PC_RECORD_SAVE_DIR = "demo_resource/2024-05-18 14-21-07.pkl" # BLUE
    # PC_RECORD_SAVE_DIR = "demo_resource/2023-06-09 14-04-12.pkl"
    PC_RECORD_FRAMES = None # 点云录制帧数

# 5.1 注意，比赛时这里应该全为False或者None
else: # default
    home_test = False
    using_video = False
    VIDEO_PATH = None
    using_dq = False
    # using_dq = True
    PC_RECORD_SAVE_DIR = None
    # PC_RECORD_SAVE_DIR = "demo_resource/2023-06-09 14-04-12.pkl"
    PC_RECORD_FRAMES = 6000 # FPS 10 录制 600S
    # PC_RECORD_FRAMES = None
    
##### advanced 测试参 #####

final_game = True # 国赛车辆阵容：5号是否后处理
alliance_game = False # 高校联盟
uv_alarm: bool = False # 像素预警
coor_alarm: bool = not uv_alarm # 雷达坐标预警
whiteBalance: bool = False # 白平衡，但牺牲帧率
record_fps = 25 # 主相机录制帧率

roi: bool = True # 主画面截取代替外置相机
if roi:
    twoCam: bool = False # 双相机
    thirdCam: bool = False # 三相机
else:
    twoCam: bool = True # 双相机
    thirdCam: bool = True # 三相机


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
|                       重要参数调整结束                            |
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# YOLO网络权重
# 雷达站检测的基本是小目标，所以送进网络的图片分辨率建议稍微大一些, 这个大小要和训练的模型匹配
# 出现cuda非法访问可能是正在炼丹造成，把imgsz改为640可正常调试
# 这两个size没有什么用
net_half = True
net_size_car = 1280  
NET_PATH_CAR = ["ultralytics/car_640_v8s_train21.engine","ultralytics/car_640_v8s_train21.engine"]
net_size_number = 640
NET_PATH_NUMBER = ["ultralytics/armor_320_v8s_train24.engine","ultralytics/armor_320_v8s_train24.engine"]
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_v5m_1_1280_int8_half.engine"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_v5m_1_1280.pt"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_7_1280_int8_half.engine"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_7_1280.pt"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_6_960_int8_half.engine"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_6_1280_int8_half.engine"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_6_1280.pt"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_5_1280_int8_half.engine"
# NET_PATH = "yolov5_61/myData/weights/roco_sjtu_zju_5_1280.pt"
YAML_PATH_CAR = "ultralytics/datasets/mydata.yaml"  # 不用autobacken的话这玩意没什么用
YAML_PATH_NUMBER = ""
# sudo 密码
PASS_WORD = "helloworld"
# 串口
USB: str = '/dev/ttyUSB0'
# 是否启动多目标跟踪
# TRACK: bool = False

# 5.6 左右相机是否需要根据各自的roi世界坐标修改？
# 相机画面中心
if not reverse:
    CAM_CENTER_L = np.array([16.8, -2.4, 1, 1])
    CAM_CENTER_R = np.array([15, -13.5, 0.5, 1])
    CAM_CENTER_BASE = np.array([26.2, -5.5, 1.1+0.3, 1])
    CAM_CENTER_BASE_VIEW =  np.array([26.2, -5.5, 1.1, 1])
else: #（中心对称，由红方进行坐标变换得到：x = 28-x；y = -15-y）
    CAM_CENTER_L = np.array([28-16.8, -15+2.4, 1, 1])
    CAM_CENTER_R = np.array([28-15, -15+13.5, 0.5, 1])
    CAM_CENTER_BASE = np.array([28-26.2, -15+5.5, 1.1+0.3, 1])
    CAM_CENTER_BASE_VIEW =  np.array([28-26.2, -15+5.5, 1.1, 1])
    
RATIO = (530/424) # ui window 长宽比
ROI_SIZE = [1060, int(1060/RATIO)]
ROI_BASE_SIZE = [int(700*1.2), int(300*1.2)]
ROI_BASE_VIEW_SIZE = [1280, 980]

# 相机掉线重启间隔(S)
CAM_RESTART_TIME_INTERVAL = 5
# 场地尺寸
ARER_LENGTH = 28
AREA_WIDTH = 15

# 位姿保存路径
LOCATION_SAVE_DIR = "pose_save"

# 小地图图片路径
MAP_PATH = "qt_design_tools/init_map.jpg"

# UI中主视频源初始图像路径
INIT_FRAME_PATH = "qt_design_tools/init_frame.jpg"
INIT_FRAME_2_PATH = "qt_design_tools/init_frame_2.jpg"
INIT_FRAME_3_PATH = "qt_design_tools/init_frame_3.jpg"
INIT_FRAME_4_PATH = "qt_design_tools/init_frame_4.jpg"

# 主相机视频保存路径
VIDEO_SAVE_DIR = "record_competition"

# 日志保存路径
LOG_DIR = "log_info"
# 相机默认参数文件位置
CAMERA_CONFIG_DIR = "Camera"
CAMERA_CONFIG_SAVE_DIR = "Camera/in_battle_config"# 相机运行中缓存参数文件位置（运行中崩溃读取参数来源）
CAMERA_YAML_PATH = "Camera_lidar_info"# 相机标定文件位置 .yaml
# 录制点云保存位置
PC_STORE_DIR = "point_record"
# 节点名称
LIDAR_TOPIC_NAME = "/livox/lidar"

# 预警信息结构体
class Alarm_result:
    def __init__(self):
        self.id = 0
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.alarm = False
        self.area = 0
Car_Info = [] # 敌方车辆坐标信息与预警信息结果列表
for i in range(7):
    Car_Info.append(Alarm_result())

# 己方坐标发送
class Our_Coor_result:
    def __init__(self):
        self.id = 0
        self.x = 0.
        self.y = 0.
        self.z = 0.
Our_Car_Info = [] # 己方车辆坐标信息与预警信息结果列表
for i in range(7):
    Our_Car_Info.append(Our_Coor_result())

# 坐标变换矩阵
# 相机到世界变换矩阵T_c2w
# 世界到相机变换矩阵T_w2c

# 5.2 左中右三个相机
T_w2c_0 = np.eye(4)
T_c2w_0 = np.linalg.inv(T_w2c_0) # 逆矩阵
T_w2c_1 = np.eye(4)
T_c2w_1 = np.linalg.inv(T_w2c_1) # 逆矩阵
T_w2c_2 = np.eye(4)
T_c2w_2 = np.linalg.inv(T_w2c_2) # 逆矩阵

armor_list = ['R1','R2','R3','R4','R5','RS','B1','B2','B3','B4','B5','BS'] # 雷达站实际考虑的各个装甲板类
unit_list = ['R1','R2','R3','R4','R5','RS','RO','RB','B1','B2','B3','B4','B5','BS','BO','BB'] # 赛场上各个目标，主要用于HP显示  # 多了前哨站和基地？
enemy2color = ['red','blue']
red = 0
blue = 1
grey = 2

# 位姿估计参考点世界坐标
# 5.6 针对本赛季地图进行修改
location_targets = {
    'home_test': # 家里测试，填自定义类似于赛场目标的空间位置
    {
        # 测试房间
        # 'blue_outpost': [9.58, -0.84, 1.36],  # 01 not reverse 哨兵轨道左上
        # 'blue_base': [9.58, -4.7, 1.36],  # 02 not reverse 哨兵轨道右上
        # 'r_rt': [3.88, -2.75, 0.35],  # 03 not reverse 斜坡下
        # 'r_lt': [5.13, -1.89, 0.73],  # 04 not reverse 斜坡上

        # 'red_outpost': [3.96, -4.37, 1.98],  # 01 reverse 货架上
        # 'red_base': [5.13, -1.89, 0.73],  # 02 reverse 斜坡上
        # 'b_rt': [3.88, -2.75, 0.35],  # 03 reverse 斜坡下
        # 'b_lt': [3.96, -4.37, 0]  # 04 reverse 货架下
        # 'red_outpost': [0, -4.00, 1.70],  # 01 reverse 墙左上
        # 'red_base': [0, -0.65, 1.70],  # 02 reverse 墙右上
        # 'b_rt': [5.13, -1.89, 0.73],  # 03 reverse 斜坡上
        # 'b_lt': [3.88, -2.75, 0.35]  # 04 reverse 斜坡下

        # 联盟赛
        'blue_outpost': [4.00, -3.25, 0],  # 01 not reverse 对面蓝左前角点
        'blue_base': [3.25, -3.25, 0],  # 02 not reverse 对面蓝左下角点
        'r_rt': [1.75, -1.75, 0],  # 03 not reverse 我方红右前角点
        'r_lt': [1.75, -0.75, 0],  # 04 not reverse 我方红左前角点

        'red_outpost': [1.00, -1.75, 0],  # 01 reverse 对面红左前角点
        'red_base': [1.75, -1.75, 0],  # 02 reverse 对面红左下角点
        'b_rt': [3.25, -3.25, 0],  # 03 reverse 我方蓝左前角点
        'b_lt': [3.25, -4.00, 0]  # 04 reverse 我方蓝右前角点
    },
     
    'game': # 红方停机坪原点笛卡尔坐标系
    {
        'blue_outpost': [16.645, -2.406, 0.118 + 1.331],  # 01 blue outpost
        'blue_base': [26.180, -7.520, 0.2 + 1.043],  # 02 blue base
        'r_rt': [8.670, -5.715 - 0.400, 0.420],  # 03 r0 right_top
        'r_lt': [8.670, -5.715, 0.420],  # 04 r0 left_top

        'red_outpost': [11.362, -12.544, 0.118 + 1.331],  # 01 red outpost
        'red_base': [1.863, -7.520, 0.2 + 1.043],  # 02 red base
        'b_rt': [19.330, -9.285 + 0.400, 0.420],  # 03 b0 right_top
        'b_lt': [19.330, -9.285, 0.420]  # 04 b0 left_top
    }
}

# 预警区域世界坐标（世界坐标系下坐标，最后一位1是方便齐次坐标系使用）
# 左上点开始逆时针凸四边形预警区域
# 调用坐标方法：print(alarm_region['area1'][0])
if home_test:
    if not reverse: # 红方预警区域（坐标为世界坐标系下齐次坐标，红方停机坪原点，对方+，左边+y）
        # 测试房间
        # alarm_area_dafu = np.array([[5, -3, 0, 1], [5, -4, 0, 1], [4, -4, 0, 1], [4, -3, 0, 1]]) # 01
        # alarm_area_feipo = np.array([[6, -3, 0, 1], [6, -4, 0, 1], [5.5, -4, 0, 1], [5.5, -3, 0, 1]]) # 02
        # alarm_area_huangao_l = np.array([[7, -3, 0, 1], [7, -4, 0, 1], [6.5, -4, 0, 1], [6.5, -3, 0, 1]]) # 03
        # alarm_area_huangao_r = np.array([[9, -3, 0, 1], [9, -4, 0, 1], [8, -4, 0, 1], [8, -3, 0, 1]]) # 04
        # 联盟赛
        alarm_area_dafu = np.array([[1.75, -1, 0, 1], [1.75, -1.75, 0, 1], [1, -1.75, 0, 1], [1, -1, 0, 1]]) # 01
        alarm_area_feipo = np.array([[5, 0, 0, 1], [5, -1, 0, 1], [4, -1, 0, 1], [4, 0, 0, 1]]) # 02
        alarm_area_huangao_l = np.array([[1, -4, 0, 1], [1, -5, 0, 1], [0, -5, 0, 1], [0, -4, 0, 1]]) # 03
        alarm_area_huangao_r = np.array([[1, 0, 0, 1], [1, -1, 0, 1], [0, -1, 0, 1], [0, 0, 0, 1]]) # 04
    else: # 蓝方预警区域（中心对称，由红方进行坐标变换得到：x = 28-x；y = -15-y）
        # 测试房间
        # alarm_area_dafu = np.array([[28-5, -15+3, 0, 1], [28-5, -15+4, 0, 1], [28-4, -15+4, 0, 1], [28-4, -15+3, 0, 1]]) # 01
        # alarm_area_feipo = np.array([[28-6, -15+3, 0, 1], [28-6, -15+4, 0, 1], [28-5.5, -15+4, 0, 1], [28-5.5, -15+3, 0, 1]]) # 02
        # alarm_area_huangao_l = np.array([[28-7, -15+3, 0, 1], [28-7, -15+4, 0, 1], [28-6.5, -15+4, 0, 1], [28-6.5, -15+3, 0, 1]]) # 03
        # alarm_area_huangao_r = np.array([[28-9, -15+3, 0, 1], [28-9, -15+4, 0, 1], [28-8, -15+4, 0, 1], [28-8, -15+3, 0, 1]]) # 04
        # 联盟赛
        alarm_area_dafu = np.array([[5-1.75, -5+1, 0, 1], [5-1.75, -5+1.75, 0, 1], [5-1, -5+1.75, 0, 1], [5-1, -5+1, 0, 1]]) # 01
        alarm_area_feipo = np.array([[5-5, -5-0, 0, 1], [5-5, -5+1, 0, 1], [5-4, -5+1, 0, 1], [5-4, -5-0, 0, 1]]) # 02
        alarm_area_huangao_l = np.array([[5-1, -5+4, 0, 1], [5-1, -5+5, 0, 1], [5-0, -5+5, 0, 1], [5-0, -5+4, 0, 1]]) # 03
        alarm_area_huangao_r = np.array([[5-1, -5+0, 0, 1], [5-1, -5+1, 0, 1], [5-0, -5+1, 0, 1], [5-0, -5-0, 0, 1]]) # 04
    alarm_region = \
            {
                'alarm_area_dafu':alarm_area_dafu,
                'alarm_area_feipo':alarm_area_feipo,
                'alarm_area_huangao_l':alarm_area_huangao_l,
                'alarm_area_huangao_r':alarm_area_huangao_r
            }
else: # 国赛   5.7 根据今年分区赛规则进行修改
    if not reverse: # 红方预警区域
        alarm_area_dafu = np.array([[19.93, -1.29, 0.85, 1], [18.71, -1.29, 0.85, 1], [18.71, -2.54, 0.85, 1], [19.93, -2.54, 0.85, 1]]) # 打符
        alarm_area_feipo = np.array([[20.68, 0, 0.15, 1], [15.37, 0, 0.15, 1], [15.37, -1.14, 0.15, 1], [20.68, -1.14, 0.15, 1]]) # 飞坡 5.7 适当宽一些无所谓
        alarm_area_huangao_l = np.array([[10.29, -6.17, 0.6, 1], [8.82, -5.71, 0.6, 1], [8.82, -8.20, 0.6, 1], [10.29, -7.73, 0.6, 1]]) # 环高左斜坡 顺序应该是从左上逆时针到右上
        # alarm_area_huangao_r = np.array([[10.61, -8.29, 0.6, 1], [9.54, -9.04, 0.6, 1], [10.83, -10.87, 0, 1], [11.89, -10.12, 0, 1]]) # 环高右斜坡
        alarm_area_dafu_our = np.array([[28- 19.93, -15+1.29, 0.85, 1], [28- 18.71, -15+1.29, 0.85, 1], [28- 18.71, -15+2.54, 0.85, 1], [28- 19.93, -15+2.54, 0.85, 1]]) # 打符
    else: # 蓝方预警区域（由红方进行坐标变换得到）
        alarm_area_dafu = np.array([[28- 19.93, -15+1.29, 0.85, 1], [28- 18.71, -15+1.29, 0.85, 1], [28- 18.71, -15+2.54, 0.85, 1], [28- 19.93, -15+2.54, 0.85, 1]]) # 打符
        alarm_area_feipo = np.array([[28- 20.68, -15+0, 0.15, 1], [28- 15.37, -15+0, 0.15, 1], [28- 15.37, -15+1.14, 0.15, 1], [28- 20.68, -15+1.14, 0.15, 1]]) # 飞坡
        alarm_area_huangao_l = np.array([[28- 10.29, -15+6.17, 0.6, 1], [28- 8.82, -15+5.71, 0.6, 1], [28- 8.82, -15+8.20, 0.6, 1], [28- 10.29, -15+7.73, 0.6, 1]]) # 环高左斜坡
        # alarm_area_huangao_r = np.array([[28- 10.61, -15+8.29, 0.6, 1], [28- 9.54, -15+9.04, 0.6, 1], [28- 10.83, -15+10.87, 0, 1], [28- 11.89, -15+10.12, 0, 1]]) # 环高右斜坡
        alarm_area_dafu_our = np.array([[19.93, -1.29, 0.85, 1], [18.71, -1.29, 0.85, 1], [18.71, -2.54, 0.85, 1], [19.93, -2.54, 0.85, 1]]) # 打符
    alarm_region = \
            {
                'alarm_area_dafu':alarm_area_dafu,
                'alarm_area_feipo':alarm_area_feipo,
                'alarm_area_huangao_l':alarm_area_huangao_l,
                'alarm_area_huangao_r':alarm_area_dafu_our
            }
            

# 发送预警消息区域优先级与预警区域唯一id
if home_test:
    loc2priority = {
            'alarm_area_feipo': 0,
            'alarm_area_dafu': 1,
            'alarm_area_huangao_l': 2,
            "alarm_area_huangao_r": 3,
    }
else:
    loc2priority = {
            'alarm_area_feipo': 0,
            'alarm_area_dafu': 1,
            'alarm_area_huangao_l': 2,
            "alarm_area_huangao_r": 3,
    }

# 发送预警消息区域接收单位编号
if home_test:
    loc2car = {
        'alarm_area_feipo': [1, 3, 4, 5],
        'alarm_area_dafu': [1, 3, 4, 5],
        'alarm_area_huangao_l': [1, 3, 4, 5, 7],
        'alarm_area_huangao_r': [1, 3, 4, 5, 7],
    }
else:
    loc2car = {
        'alarm_area_feipo': [1, 3, 4, 5],
        'alarm_area_dafu': [1, 3, 4, 5],
        'alarm_area_huangao_l': [1, 3, 4, 5, 7],
        'alarm_area_huangao_r': [1, 3, 4, 5, 7],
    }

color2enemy = {"red":0, "blue":1}


# 5.3 第一个是中间mindvision，后面两个海康相机
camera_match_list = \
    ['042030120299',
     'DA2357361',  
     'DA2357357']

# 海康右相机 DA2357361  (标签上写着)
# 海康左相机 DA2357357

# 调参界面位置参数
preview_location = [
    (100, 100), (940, 100)
]  # 主相机使用第一个位置，左右相机使用第二个位置

# 白平衡
wb_bar_max = 100
wb_param = [0, 0, 0]
