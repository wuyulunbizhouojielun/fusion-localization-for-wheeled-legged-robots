'''
default network class
给神经网络类的接口格式定义，神经网络具体需要自行添加
'''

# 4.21 对装甲板车辆匹配去重进行优化，主要是本文件不包含track部分

import numpy as np

from radar_module.config import red, blue, grey, final_game
# from yolov5_6.mydetect import network_init, run_detect
from ultralytics.engine.mypredictor import network_init, run_detect
from radar_module.common import is_inside

from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import scale_coords
from functools import cmp_to_key # 结构体排序

# 4.27 更新为第一层1280v8s，第二层640v8m

class Detect_result:
    '''
    yolov5检测结果结构体
    '''
    def __init__(self):
        self.id = 0
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.conf = 0.

def sort_by_conf(a, b):
    '''
    定义排序函数  降序
    '''
    if a.conf >  b.conf:
        return -1 # 不调换
    else:
        return 1 # 调换

class Final_result:
    '''
    去重结果结构体，包含敌我所有单位
    id:0-car;1-watcher
    color:0-red;1-blue
    num:1~5
    '''
    def __init__(self):
        self.id = 0
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.color = grey
        self.num = 0
        self.armor_x1 = 0.
        self.armor_y1 = 0.
        self.armor_x2 = 0.
        self.armor_y2 = 0.
        self.track_id = None



class Predictor(object):
    def __init__(self, weights_CAR="", classes_CAR="", weights_NUMBER="", classes_NUMBER=""):
        '''
        :param weights:.pt文件的存放地址
        '''
        # self._half = half
        self._weights_car = weights_CAR
        self._weights_number = weights_NUMBER
        self._classes_car = classes_CAR
        self._classes_number = classes_NUMBER
        # self._model, self._names = network_init(weights=self._weights, data=classes, half=self._half)
        self._model_car, self._model_number = network_init(weights_car=self._weights_car, weights_number=self._weights_number)
        self._names=['car','RS','R1','R2','R3','R4','R5','BS','B1','B2','B3','B4','B5','GS','G1','G2','G3','G4','G5','ignore']
        self._pred = [None, None]


    def infer(self, input_img, camera_type, open_track = False):
        '''
        这个函数用来预测

        :param input_img:输入网络的图片
        '''
        
        # 旁边小相机换小的car和armor模型
        if camera_type == 0:
            self._pred[0] =  run_detect(image=input_img, model_car=self._model_car[0], model_number=self._model_number[0], conf_thres=0.3, iou_thres=0.45, do_track=open_track, camera_type = camera_type)
        elif camera_type == 1:
            self._pred[1] =  run_detect(image=input_img, model_car=self._model_car[1], model_number=self._model_number[1], conf_thres=0.3, iou_thres=0.45, do_track=open_track, camera_type = camera_type)
        # elif camera_type == 2:
        #     self._pred[2] =  run_detect(image=input_img, model_car=self._model_car[1], model_number=self._model_number[1], conf_thres=0.3, iou_thres=0.45, do_track=open_track, camera_type = camera_type)
    
    def network_result_process(self):
        # 5.2 对预测不到任何结果的情况进行分类
        # 5.3 三个相机去重，物理上还是比较难保证区域不重复
        # 5.4 如果车都没识别到装甲板，这样子并不行，测试的时候是极端情况三个画面完全一样
        # 跟踪的话发生在预测阶段，所以每个相机分配的id独立，track_id_cls修改为列表才行
        FinalResult = []
        car_number = 0
        if self._pred[0] is None and self._pred[1] is None:
            return car_number, FinalResult
        
        temp = self._pred[0] + self._pred[1] # 测试了下，有重复的也会保留
        # temp = self._pred[0]  # 5.13 测试特化

        # 对armor_label相同的结构体进行去重
        key_list = ['RS', 'R1', 'R2', 'R3', 'R4', 'R5', 'BS', 'B1', 'B2', 'B3', 'B4', 'B5']
        num_cls_index = {'0':[], 'RS':[], 'R1':[], 'R2':[], 'R3':[], 'R4':[], 'R5':[], 'BS':[], 'B1':[], 'B2':[], 'B3':[], 'B4':[], 'B5':[]}
        
        # 存储各个类别对应的索引数量，索引大于1需要去重
        # 去重不能直接删除，因为索引会变化，索引里保留置信度高的，之后统一添加到 FinalResult
        for k in range(0, len(temp)):
            armor_id = temp[k].id
            if armor_id in key_list:
                # 如果为空则直接添加，为未知类别也直接保留
                if (not len(num_cls_index[armor_id])) or armor_id == '0':
                    num_cls_index[armor_id].append(k)
                # 如果某个标签列表长度不是0，比较之后添加置信度高的，即除了'0'以外其他的列表长度都不超过1
                else:
                    index = num_cls_index[armor_id][0]
                    if temp[k].armor_conf <= temp[index].armor_conf:
                        continue
                    else:
                       num_cls_index[armor_id][0] = k   # 更新索引

        # 按照索引值整理
        for key, value in num_cls_index.items():
            for i in range(0, len(value)):
                FinalResult.append(temp[value[i]])     
                    
        car_number = len(FinalResult)
        return car_number, FinalResult
            
        