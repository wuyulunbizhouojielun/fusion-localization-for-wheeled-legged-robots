import numpy as np

import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

# 本文件负责读取训练好的yolov8模型，并将其用于目标检测任务，替换老代码里面的mydetect.py

################目录设置，根目录更改为yolov8###############
# root directory setting
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
print("YOLOv8 root directory is %s\r\n" % (ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# relative directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
print("YOLOv8 relative root directory is %s\r\n" % (ROOT))

from ultralytics.data.augment import LetterBox
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.data.loaders import LoadImages, LoadStreams
from ultralytics.utils.ops import (LOGGER, non_max_suppression, scale_coords, xyxy2xywh)
from ultralytics.cfg.__init__ import colorstr
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import select_device, time_sync,strip_optimizer
from ultralytics.utils.checks import print_args,check_file, check_imshow, check_requirements
from ultralytics.utils.files import increment_path


# 存储每一帧的track_id以及对应的cls
# 5.6 这个全局变量只在这个文件有用，我觉得在整个主程序周期内都是存在的，试了一下mian里面import的话也能看得到一直有修改
# 5.6 三相机追踪的话得用三个字典，否则序号难免重复
trackid_cls_index = [{},{},{}]

# 追踪标签筛选
# 5.6 三个相机视角分三个追踪器，还有python修改字典内的值还是得用序号改，直接遍历不能修改
def trackid_filter(track_id_now, cls_now, camera_type):
    global trackid_cls_index
    modified = False
    for key,value in trackid_cls_index[camera_type].items():
        # 如果另一个trackid的机器人标签和现在这个id的相同，则将该机器人标为未知类别
        if value == cls_now and key !=track_id_now:
            trackid_cls_index[camera_type][key] = '0'
            modified = True
    return modified

# 网络初始化
def network_init(weights_car,  # model.pt path
                 weights_number
                 ):
    # Initialize
    # device = select_device(device)
    # model_temp = AutoBackend(weights=weights,device=device,data=data)   # 3.4 用于推导, 仅仅用于确定类型，感觉不够优雅  3.5 或者就用enumerate出来的det获取names, 不过这个autobacken也就初始化的时候执行，感觉影响并不大
    # print(model.info())
    # stride, names, pt = model.stride, model.names, model.pt
    
    model_car = YOLO(weights_car) # 3.3 是否还有其他可选参数？ 3.4 似乎没有, 模型精度感觉可以在导出的时候设置
    model_number = YOLO(weights_number)
    return model_car, model_number

# 预测结果封装
# 4.19 现在预测阶段直接确定car对应的id(不是cls，cls是0，id是车辆的标签)
# 双相机暂时不考虑开track
# class pred_results:
#     def __init__(self, xyxy, cls, car_conf):
#         self.xyxy = xyxy
#         self.cls = cls
#         self.car_conf = car_conf
#         self.camera_type = camera_type
#         self.id = None
#         self.track_id = None


# 4.21
class Final_result:
    '''
    去重结果结构体，包含敌我所有单位
    '''
    def __init__(self):
        self.id = '0'    # str类型 如 R1 默认car无装甲板或者G类为0
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.color = 2     # 默认 gray 2
        self.num = 0  # int 类型
        self.armor_x1 = 0.
        self.armor_y1 = 0.
        self.armor_x2 = 0.
        self.armor_y2 = 0.
        self.car_conf = 0.
        self.armor_conf = 0.
        self.track_id = None
        self.camera_type = 0


# yolov8运行检测
def run_detect(image,  # img
               model_car,  # initialized model
               model_number,
               conf_thres=0.3,  # confidence threshid_match_indexold
               iou_thres=0.45,  # NMS IOU threshold
               do_track = False,
               camera_type = 0
               ):

    # 分两阶段进行目标检测
    
    # 已匹配标识符，匹配到的键值取1，防止标签重复
    id_match_index = {'RS': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'BS': 0, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0}
    # 数字类别和id映射字典 G可去掉
    number_str_index = {'0':'0', '1':'RS', '2':'R1', '3':'R2', '4':'R3', '5':'R4', '6':'R5', '7':'BS', '8':'B1', '9':'B2', '10':'B3', '11':'B4', '12':'B5'}
    # id 类别和num映射字典 G可去掉
    id_num_index = {'RS': 6, 'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4, 'R5': 5, 'BS':6, 'B1': 1, 'B2':2, 'B3':3, 'B4': 4, 'B5': 5}
    # 根据标签确定颜色
    red_id = ['RS', 'R1', 'R2', 'R3', 'R4', 'R5']
    blue_id = ['BS', 'B1', 'B2', 'B3', 'B4', 'B5']

    # car检测
    pred = []  # 5.6 FinalResult列表
    k = 0  # pred[]内元素序号
    
    # 5.6 关于tensor类型转换后是否共享内存，item()方法返回的是standard python 
    # 第一帧track_id列表为空，单独处理
    if do_track:
        results_car = model_car.track(image, imgsz=1280, augment=False, visualize=False, conf = conf_thres)
        for i, r_car in enumerate(results_car):
            if r_car.boxes.id is None:
                for car_box, car_cls, car_conf in zip(r_car.boxes.xyxy, r_car.boxes.cls, r_car.boxes.conf):            # 如果有多辆车，每次返回一辆车的xyxy
                    car_roi = image[int(car_box[1]):int(car_box[3]), int(car_box[0]):int(car_box[2])]
                    pred.append(Final_result()) # 因为初始化不知道大小，所以得先append
                    # 只包含一个元素的张量用.item()，列表用tolist()，返回数字
                    pred[k].x1 = float(car_box[0].item())
                    pred[k].y1 = float(car_box[1].item())
                    pred[k].x2 = float(car_box[2].item())
                    pred[k].y2 = float(car_box[3].item())
                    pred[k].car_conf = float(car_conf.item())
                    pred[k].camera_type = camera_type
                    # 装甲板检测
                    results_number = model_number.predict(car_roi, imgsz=640, augment=False, visualize=False)
                    max_number_conf = 0.0
                    number_box_temp = [0.0, 0.0, 0.0, 0.0] 
                    number_cls_temp = 0
                    for j, r_number in enumerate(results_number):
                        # 修改坐标，因为裁剪出来的图片不在原来的坐标系下
                        for number_box, number_cls, number_conf in zip(r_number.boxes.xyxy, r_number.boxes.cls, r_number.boxes.conf):
                            # 遍历所有number的检测结果，选取出置信度最高的作为car的id
                            number_conf_temp = float(number_conf.item())
                            if number_conf_temp > max_number_conf:
                                # 更新置信度更高的结果
                                max_number_conf = number_conf_temp
                                # 由于共享内存的问题，不能直接修改预测结果，所以用新的变量替代
                                # 坐标映射回原来的坐标
                                number_box_temp[0] = (float(number_box[0].item()+car_box[0].item()))
                                number_box_temp[1] = (float(number_box[1].item()+car_box[1].item()))
                                number_box_temp[2] = (float(number_box[2].item()+car_box[0].item()))
                                number_box_temp[3] = (float(number_box[3].item()+car_box[1].item()))
                                # 根据self._names标签更改装甲板对应标签，可以通过训练时重新加入car或者随便找一个东西凑数解决
                                number_cls_temp = int(number_cls.item() + 1)   # 后移一个位置，装甲板从1开始，注意直接出来的是小数
                                number_cls_temp = str(number_cls_temp)
                    
                    # 4.27 如果检测不到装甲板，需要保证一定是字符串，而且检测不到装甲板的话这里就直接跳过
                    id_temp = number_str_index[str(number_cls_temp)]
                    if id_temp == '0':
                        continue
                    
                    if not id_match_index[id_temp]:  # 尚未匹配
                        pred[k].id = id_temp
                        pred[k].num = id_num_index[id_temp]
                        pred[k].armor_conf = max_number_conf
                        pred[k].armor_x1 = number_box_temp[0]
                        pred[k].armor_y1 = number_box_temp[1]
                        pred[k].armor_x2 = number_box_temp[2]
                        pred[k].armor_y2 = number_box_temp[3]
                        if id_temp in red_id:
                            pred[k].color = 0   # config 文件内， red 0 blue 1 gray 2
                        elif id_temp in blue_id:
                            pred[k].color = 1
                        id_match_index[id_temp] = 1

                    k += 1
                
            elif r_car.boxes.id is not None:
                for car_box, car_cls, car_conf, car_id in zip(r_car.boxes.xyxy, r_car.boxes.cls, r_car.boxes.conf, r_car.boxes.id):         
                    car_roi = image[int(car_box[1]):int(car_box[3]), int(car_box[0]):int(car_box[2])]
                    pred.append(Final_result()) # 因为初始化不知道大小，所以得先append
                    # 只包含一个元素的张量用.item()，列表用tolist()，返回数字
                    pred[k].x1 = float(car_box[0].item())
                    pred[k].y1 = float(car_box[1].item())
                    pred[k].x2 = float(car_box[2].item())
                    pred[k].y2 = float(car_box[3].item())
                    pred[k].car_conf = float(car_conf.item())
                    pred[k].track_id = int(car_id.item())
                    pred[k].camera_type = camera_type
                    # 装甲板检测
                    results_number = model_number.predict(car_roi, imgsz=640, augment=False, visualize=False)
                    max_number_conf = 0.0
                    number_box_temp = [0.0, 0.0, 0.0, 0.0] 
                    number_cls_temp = 0
                    for j, r_number in enumerate(results_number):
                        # 修改坐标，因为裁剪出来的图片不在原来的坐标系下
                        for number_box, number_cls, number_conf in zip(r_number.boxes.xyxy, r_number.boxes.cls, r_number.boxes.conf):
                            # 遍历所有number的检测结果，选取出置信度最高的作为car的id
                            number_conf_temp = float(number_conf.item())
                            if number_conf_temp > max_number_conf:
                                # 更新置信度更高的结果
                                max_number_conf = number_conf_temp
                                # 由于共享内存的问题，不能直接修改预测结果，所以用新的变量替代
                                # 坐标映射回原来的坐标
                                number_box_temp[0] = (float(number_box[0].item()+car_box[0].item()))
                                number_box_temp[1] = (float(number_box[1].item()+car_box[1].item()))
                                number_box_temp[2] = (float(number_box[2].item()+car_box[0].item()))
                                number_box_temp[3] = (float(number_box[3].item()+car_box[1].item()))
                                # 根据self._names标签更改装甲板对应标签，可以通过训练时重新加入car或者随便找一个东西凑数解决
                                number_cls_temp = int(number_cls.item() + 1)   # 后移一个位置，装甲板从1开始，注意直接出来的是小数
                                number_cls_temp = str(number_cls_temp)
                    
                    # 4.27 如果检测不到装甲板，需要保证一定是字符串，而且检测不到装甲板的话这里就直接跳过
                    id_temp = number_str_index[str(number_cls_temp)]
                    if id_temp == '0':
                        continue
                    
                    if not id_match_index[id_temp]:  # 尚未匹配
                        pred[k].id = id_temp
                        pred[k].num = id_num_index[id_temp]
                        pred[k].armor_conf = max_number_conf
                        pred[k].armor_x1 = number_box_temp[0]
                        pred[k].armor_y1 = number_box_temp[1]
                        pred[k].armor_x2 = number_box_temp[2]
                        pred[k].armor_y2 = number_box_temp[3]
                        if id_temp in red_id:
                            pred[k].color = 0   # config 文件内， red 0 blue 1 gray 2
                        elif id_temp in blue_id:
                            pred[k].color = 1
                        id_match_index[id_temp] = 1

                    k += 1
    
        # 针对每一帧的track结果，对装甲板标签进行更新
        car_num = k
        global trakc_id_cls_index
        update = True
        if not len(trackid_cls_index[camera_type]):
            for i in range(car_num):
                trackid_cls_index[camera_type].update({str(pred[i].track_id): pred[i].id})  
        # 之前没匹配到的car的id感觉应该是'0' ，可以调试看看灰框的时候
        else:
            for i in range(car_num):
                # 查询id
                # 每次修改trackid_cls_index字典内的value，必须确保标签唯一，优先相信新检测到的标签
                if pred[i].track_id is not None:
                    track_id_now = str(pred[i].track_id)
                    if track_id_now in trackid_cls_index[camera_type].keys():
                        if trackid_cls_index[camera_type][track_id_now] == '0' and pred[i].id != '0':
                            modified = trackid_filter(track_id_now, pred[i].id, camera_type)
                            trackid_cls_index[camera_type].update({track_id_now: pred[i].id}) # 字典更新很智能，已有的key会自动更新value
                        elif trackid_cls_index[camera_type][track_id_now] != '0' and pred[i].id != '0':
                            modified = trackid_filter(track_id_now, pred[i].id, camera_type)
                            trackid_cls_index[camera_type].update({track_id_now: pred[i].id})
                        elif trackid_cls_index[camera_type][track_id_now] != '0' and pred[i].id == '0':
                            # 和目前所有检测结果比较，防止重复
                            for j in range(car_num):
                                if trackid_cls_index[camera_type][track_id_now] == pred[j].id:
                                    update = False
                                    break
                            if update:
                                pred[i].track_id = trackid_cls_index[camera_type][track_id_now]  # 保持之前检测到的标签
                        else:
                            continue
                    else:
                        # 新出现的id
                        modified = trackid_filter(track_id_now, pred[i].id, camera_type)
                        trackid_cls_index[camera_type].update({track_id_now: pred[i].id})
                
                # 到后期跟踪的东西可能越来越多，之前的有些id估计用不到了，感觉得清理一下
                # 或者之后可以优化成队列，这样就不会出现一下子全部跟踪丢失
                if len(trackid_cls_index[camera_type]) > 60:
                    trackid_cls_index[camera_type].clear()
        
                
    else:
        results_car = model_car.predict(image, imgsz=1280, augment=False, visualize=False, conf = conf_thres)
        for i, r_car in enumerate(results_car):
            for car_box, car_cls, car_conf in zip(r_car.boxes.xyxy, r_car.boxes.cls, r_car.boxes.conf):            # 如果有多辆车，每次返回一辆车的xyxy
                car_roi = image[int(car_box[1]):int(car_box[3]), int(car_box[0]):int(car_box[2])]
                pred.append(Final_result()) # 因为初始化不知道大小，所以得先append
                # 只包含一个元素的张量用.item()，列表用tolist()，返回数字
                pred[k].x1 = float(car_box[0].item())
                pred[k].y1 = float(car_box[1].item())
                pred[k].x2 = float(car_box[2].item())
                pred[k].y2 = float(car_box[3].item())
                pred[k].car_conf = float(car_conf.item())
                pred[k].camera_type = camera_type
                # 装甲板检测
                results_number = model_number.predict(car_roi, imgsz=640, augment=False, visualize=False)
                max_number_conf = 0.0
                number_box_temp = [0.0, 0.0, 0.0, 0.0] 
                number_cls_temp = 0
                for j, r_number in enumerate(results_number):
                    # 修改坐标，因为裁剪出来的图片不在原来的坐标系下
                    for number_box, number_cls, number_conf in zip(r_number.boxes.xyxy, r_number.boxes.cls, r_number.boxes.conf):
                        # 遍历所有number的检测结果，选取出置信度最高的作为car的id
                        number_conf_temp = float(number_conf.item())
                        if number_conf_temp > max_number_conf:
                            # 更新置信度更高的结果
                            max_number_conf = number_conf_temp
                            # 由于共享内存的问题，不能直接修改预测结果，所以用新的变量替代
                            # 坐标映射回原来的坐标
                            number_box_temp[0] = (float(number_box[0].item()+car_box[0].item()))
                            number_box_temp[0] = (float(number_box[1].item()+car_box[1].item()))
                            number_box_temp[0] = (float(number_box[2].item()+car_box[0].item()))
                            number_box_temp[0] = (float(number_box[3].item()+car_box[1].item()))
                            # 根据self._names标签更改装甲板对应标签，可以通过训练时重新加入car或者随便找一个东西凑数解决
                            number_cls_temp = int(number_cls.item() + 1)   # 后移一个位置，装甲板从1开始，注意直接出来的是小数
                            number_cls_temp = str(number_cls_temp)
                
                # 4.27 如果检测不到装甲板，需要保证一定是字符串，而且检测不到装甲板的话这里就直接跳过
                id_temp = number_str_index[str(number_cls_temp)]
                if id_temp == '0':
                    continue
                
                if not id_match_index[id_temp]:  # 尚未匹配
                    pred[k].id = id_temp
                    pred[k].num = id_num_index[id_temp]
                    pred[k].armor_conf = max_number_conf
                    pred[k].armor_x1 = number_box_temp[0]
                    pred[k].armor_y1 = number_box_temp[1]
                    pred[k].armor_x2 = number_box_temp[2]
                    pred[k].armor_y2 = number_box_temp[3]
                    if id_temp in red_id:
                        pred[k].color = 0   # config 文件内， red 0 blue 1 gray 2
                    elif id_temp in blue_id:
                        pred[k].color = 1
                    id_match_index[id_temp] = 1

                k += 1

    return pred   # k 相当于car_number


# 程序入口
if __name__ == "__main__":
    model, names = network_init(weights='ultralytics/car_640.engine')
    # model, names = network_init(weights='yolov5_61/myData/weights/roco_sjtu_zju_5_1280_int8.engine')
    # model, names = network_init(weights='yolov5_61/myData/weights/roco_sjtu_zju_5_1280.pt')
    capture = cv2.VideoCapture("demo_resource/videos/sjtu.mp4")    # 从相机或视频流中获取图像

    while (True):

        flag0, frame0 = capture.read()
        pred, img = run_detect(image=frame0, model=model, imgsz=640)

        for i, det in enumerate(pred):  # detections per image
            s = ''
            annotator = Annotator(frame0, line_width=3, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += str(n.item()) + ' ' + str(names[int(c)]) + ' '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if (names[int(cls)] == "car") | (names[int(cls)] == "bus") | (names[int(cls)] == "truck"): #自定义可视化类别

                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # print(xyxy)

                    # show target center
                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())
                    centerX = int((x1 + x2) / 2)
                    centerY = int((y1 + y2) / 2)
                    class_index = cls
                    object_name = names[int(cls)]
                    cv2.circle(frame0, (centerX, centerY), 3, (255, 255, 0), 3, 0)
        
        cv2.namedWindow("frame0",cv2.WINDOW_NORMAL)
        cv2.imshow("frame0", frame0)
        key = cv2.waitKey(1)

