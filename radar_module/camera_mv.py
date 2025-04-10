'''
相机类
本脚本基于MindVision官方demo from http://www.mindvision.com.cn/rjxz/list_12.aspx?lcid=139
我们为各个相机指定了编号，
请使用者确认使用的是MindVision相机并修改本脚本中参数以适应对应的传感器安装方案
'''
from re import T
import sys
import os
# print('os.getcwd')
# print(os.getcwd())
sys.path.append(os.getcwd())
# print('sys.path')
# print(sys.path)

import cv2
import numpy as np
from mdvision_sdk import mvsdk
import yaml
import time
from datetime import datetime
from radar_module.config import camera_match_list,CAMERA_CONFIG_DIR,CAMERA_YAML_PATH,CAMERA_CONFIG_SAVE_DIR,preview_location, wb_bar_max, wb_param, whiteBalance

def open_camera(camera_type, is_init, date):
    # try to open the camera
    cap = None
    init_flag = False
    try:
        cap = HT_Camera(camera_type, is_init, date)
        r, frame = cap.read()  # read once to examine whether the cap is working
        assert r, "Camera not init"  # 读取失败则报错
        r, frame = cap.read()
        if not is_init:  # 若相机已经启动过一次则不进行调参
            # 建立预览窗口
            cv2.namedWindow("preview of {0} press T to adjust".format(camera_type), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("preview of {0} press T to adjust".format(camera_type), 840, 640)
            cv2.setWindowProperty("preview of {0} press T to adjust".format(camera_type), cv2.WND_PROP_TOPMOST, 1)
            # win_loc = 0 if camera_type in [0, 1] else 1
            win_loc = 0 if camera_type in [0] else 1
            # 移动至合适位置
            cv2.moveWindow("preview of {0} press T to adjust".format(camera_type), *preview_location[win_loc])
            cv2.imshow("preview of {0} press T to adjust".format(camera_type), frame)
            key = cv2.waitKey(0)
            cv2.destroyWindow("preview of {0} press T to adjust".format(camera_type))
            # 按其他键则不调参使用默认参数，按t键则进入调参窗口，可调曝光和模拟增益
            if key == ord('t') & 0xFF:
                cap.NoautoEx()
                # camera_type > 0 表示该相机为云台相机(cam 1,2)
                if camera_type > 0:
                    tune_exposure(cap, high_fps=True)
                else:
                    tune_exposure(cap)
            cap.saveParam(date)  # 保存参数，以便在比赛中重启时能够自动读取准备阶段设置的参数
        init_flag = True
    except Exception as e:
        # If cap is not open, hcamera will be -1 and cap.release will do nothing. If camera is open, cap.release will close the camera
        cap.release()
        print("[ERROR] {0}".format(e))
    return init_flag, cap


class Camera_Thread(object):
    def __init__(self, camera_type, video=False, video_path=None):
        '''
        the Camera_Thread class which will restart camera when it broken

        :param camera_type: 相机编号
        :param video: 是否使用视频
        :param video_path: 视频路径

        '''
        self._camera_type = camera_type
        self._date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self._open = False
        self._cap = None
        self._is_init = False
        self._video = video
        self._video_path = video_path
        # try to open it once
        self.open()

    def open(self):
        # if camera not open, try to open it
        if not self._video:
            if not self._open:
                self._open, self.cap = open_camera(self._camera_type, self._is_init, self._date)
                if not self._is_init and self._open:
                    self._is_init = True
        else:
            if not self._open:
                self.cap = cv2.VideoCapture(self._video_path)
                self._open = True
                if not self._is_init and self._open:
                    self._is_init = True

    def is_open(self):
        '''
        check the camera opening state
        '''
        return self._open

    def read(self):
        if self._open:
            r, frame = self.cap.read()
            if not r:
                self.cap.release()  # release the failed camera
                self._open = False
            return r, frame
        else:
            return False, None

    def release(self):
        if self._open:
            self.cap.release()
            self._open = False

    def __del__(self):
        if self._open:
            self.cap.release()
            self._open = False


def read_yaml(camera_type):
    '''
    读取相机标定参数,包含外参，内参，以及关于雷达的外参

    :param camera_type:相机编号
    :return: 读取成功失败标志位，相机内参，畸变系数，和雷达外参，相机图像大小
    '''
    yaml_path = "{0}/camera{1}.yaml".format(CAMERA_YAML_PATH,
                                            camera_type)
    try:
        with open(yaml_path, 'rb')as f:
            res = yaml.load(f, Loader=yaml.FullLoader)
            K_0 = np.float32(res["K_0"]).reshape(3, 3)
            C_0 = np.float32(res["C_0"])
            E_0 = np.float32(res["E_0"]).reshape(4, 4)
            imgsz = tuple(res['ImageSize'])

        return True, K_0, C_0, E_0, imgsz
    except Exception as e:
        print("[ERROR] {0}".format(e))
        return False, None, None, None, None


class HT_Camera:
    def __init__(self, camera_type=0, is_init=True, path=None):
        '''
        相机驱动类

        :param camera_type:相机编号
        :param is_init: 相机是否已经启动过一次，若是则使用path所指向的参数文件
        :param path: 初次启动保存的参数文件路径名称（无需后缀，实际使用时即为创建时间）
        '''
        # 枚举相机
        DevList = mvsdk.CameraEnumerateDevice()
        # 得到存在相机序列号
        existing_camera_name = [dev.GetSn() for dev in DevList]

        if not camera_match_list[camera_type] in existing_camera_name:
            # 所求相机不存在
            self.hCamera = -1
            return

        camera_no = existing_camera_name.index(camera_match_list[camera_type])  # 所求相机在枚举列表中编号
        DevInfo = DevList[camera_no]
        print("{} {}".format(DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        print(DevInfo)

        self.camera_type = camera_type

        # 打开相机
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            self.hCamera = -1
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(self.hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        if not is_init:
            # default camera parameter config
            param_path = "{0}/camera_{1}.Config".format(CAMERA_CONFIG_DIR,
                                                        camera_type)

            mvsdk.CameraReadParameterFromFile(self.hCamera, param_path)
        else:
            # 初次启动后，保存的参数文件
            param_path = "{0}/camera_{1}_of_{2}.Config".format(CAMERA_CONFIG_SAVE_DIR,
                                                               camera_type, path)

            mvsdk.CameraReadParameterFromFile(self.hCamera, param_path)

        mvsdk.CameraSetAeState(self.hCamera, 0)

        print(f"[INFO] camera exposure time {mvsdk.CameraGetExposureTime(self.hCamera) / 1000:0.03f}ms")
        print(f"[INFO] camera gain {mvsdk.CameraGetAnalogGain(self.hCamera):0.03f}")

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    def read(self):
        if self.hCamera == -1:
            return False, None
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(
                (FrameHead.iHeight, FrameHead.iWidth,
                 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            return True, frame
        except mvsdk.CameraException as e:
            print(e)
            return False, None

    def setExposureTime(self, ex=30):
        if self.hCamera == -1:
            return
        mvsdk.CameraSetExposureTime(self.hCamera, ex)

    def setGain(self, gain):
        if self.hCamera == -1:
            return
        mvsdk.CameraSetAnalogGain(self.hCamera, gain)

    def saveParam(self, path):

        if self.hCamera == -1:
            return
        if not os.path.exists(CAMERA_CONFIG_SAVE_DIR):
            os.mkdir(CAMERA_CONFIG_SAVE_DIR)
        param_path = "{0}/camera_{1}_of_{2}.Config".format(CAMERA_CONFIG_SAVE_DIR, self.camera_type, path)
        mvsdk.CameraSaveParameterToFile(self.hCamera, param_path)

    def NoautoEx(self):
        '''
        设置不自动曝光
        '''
        if self.hCamera == -1:
            return
        mvsdk.CameraSetAeState(self.hCamera, 0)

    def getExposureTime(self):
        if self.hCamera == -1:
            return -1
        return int(mvsdk.CameraGetExposureTime(self.hCamera) / 1000)

    def getAnalogGain(self):
        if self.hCamera == -1:
            return -1
        return int(mvsdk.CameraGetAnalogGain(self.hCamera))

    def release(self):
        if self.hCamera == -1:
            return
        # 关闭相机
        mvsdk.CameraUnInit(self.hCamera)
        # 释放帧缓存
        mvsdk.CameraAlignFree(self.pFrameBuffer)


def tune_exposure(cap: HT_Camera, high_fps=False):
    '''
    相机参数调节
    :param cap: camera target
    :param high_fps: 采用微秒/毫秒为单位调整曝光时间
    '''
    cv2.namedWindow("exposure press q to exit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("exposure press q to exit", 1280, 960)
    cv2.moveWindow("exposure press q to exit", 10, 10)
    cv2.setWindowProperty("exposure press q to exit", cv2.WND_PROP_TOPMOST, 1)
    if high_fps:
        cv2.createTrackbar("exposure ", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("exposure ", "exposure press q to exit", 50)
        cv2.setTrackbarMin("exposure ", "exposure press q to exit", 0)
        # cv2.setTrackbarPos("exposure ", "exposure press q to exit", int(cap.getExposureTime() * 1000))
        cv2.setTrackbarPos("exposure ", "exposure press q to exit", 20) # 默认曝光时间
        # 模拟增益区间为0到256
        cv2.createTrackbar("gain", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("gain", "exposure press q to exit", 256)
        cv2.setTrackbarMin("gain", "exposure press q to exit", 0)
        # cv2.setTrackbarPos("gain", "exposure press q to exit", int(cap.getAnalogGain()))
        cv2.setTrackbarPos("gain", "exposure press q to exit", 5) # 默认增益
    else:
        cv2.createTrackbar("exposure ", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("exposure ", "exposure press q to exit", 100)
        cv2.setTrackbarMin("exposure ", "exposure press q to exit", 0)
        # cv2.setTrackbarPos("exposure ", "exposure press q to exit", int(cap.getExposureTime()))
        cv2.setTrackbarPos("exposure ", "exposure press q to exit", 20) # 默认曝光时间ms
        cv2.createTrackbar("gain", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("gain", "exposure press q to exit", 256)
        cv2.setTrackbarMin("gain", "exposure press q to exit", 0)
        # cv2.setTrackbarPos("gain", "exposure press q to exit", int(cap.getAnalogGain()))
        cv2.setTrackbarPos("gain", "exposure press q to exit", 50) # 默认增益

    # white balance
    # auto param
    if not high_fps: # 云台相机不调节白平衡
        flag, frame = cap.read()
        img, b_init, g_init, r_init = auto_white_balance_once(frame)
        b_init = int(b_init)
        g_init = int(g_init)
        r_init = int(r_init)

        cv2.createTrackbar("wb_b", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("wb_b", "exposure press q to exit", wb_bar_max)
        cv2.setTrackbarMin("wb_b", "exposure press q to exit", 0)
        cv2.setTrackbarPos("wb_b", "exposure press q to exit", b_init)
        cv2.createTrackbar("wb_g", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("wb_g", "exposure press q to exit", wb_bar_max)
        cv2.setTrackbarMin("wb_g", "exposure press q to exit", 0)
        cv2.setTrackbarPos("wb_g", "exposure press q to exit", g_init)
        cv2.createTrackbar("wb_r", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("wb_r", "exposure press q to exit", wb_bar_max)
        cv2.setTrackbarMin("wb_r", "exposure press q to exit", 0)
        cv2.setTrackbarPos("wb_r", "exposure press q to exit", r_init)

    flag, frame = cap.read()

    while (flag and cv2.waitKey(1) != ord('q') & 0xFF):
        if high_fps:
            # cap.setExposureTime(cv2.getTrackbarPos("exposure ", "exposure press q to exit"))
            cap.setExposureTime(cv2.getTrackbarPos("exposure ", "exposure press q to exit") * 1000)
        else:
            cap.setExposureTime(cv2.getTrackbarPos("exposure ", "exposure press q to exit") * 1000)
        cap.setGain(cv2.getTrackbarPos("gain", "exposure press q to exit"))

        if not high_fps:
            wb_param[0] = cv2.getTrackbarPos("wb_b", "exposure press q to exit")
            wb_param[1] = cv2.getTrackbarPos("wb_g", "exposure press q to exit")
            wb_param[2] = cv2.getTrackbarPos("wb_r", "exposure press q to exit")

        cv2.imshow("exposure press q to exit", frame)
        flag, frame = cap.read()
        if not high_fps:
            if whiteBalance:
                frame = white_balance(frame, wb_param)

    ex = cv2.getTrackbarPos("exposure ", "exposure press q to exit")
    g1 = cv2.getTrackbarPos("gain", "exposure press q to exit")
    # if high_fps:
    #     ex = ex / 1000
    print(f"finish set exposure time {ex:.03f}ms")
    print(f"finish set analog gain {g1}")
    cv2.destroyWindow("exposure press q to exit")


def auto_white_balance_once(img):
    '''
    自动白平衡，设置滑动条初值
    return: wb bar 3参数初始值
    '''
    b, g, r = cv2.split(img)
    b_avg = cv2.mean(b)[0]
    g_avg = cv2.mean(g)[0]
    r_avg = cv2.mean(r)[0]
    k = (r_avg + g_avg + b_avg) / 3# 求各个通道所占增益
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    img = cv2.merge([b, g, r])

    return img, (kb-1)*wb_bar_max, (kg-1)*wb_bar_max, (kr-1)*wb_bar_max


def white_balance(img, wb_param = wb_param):
    '''
    白平衡
    手动调节后的参数作为输入
    '''
    (b, g, r) = cv2.split(img)
    kr = 1 + (wb_param[2])/wb_bar_max
    kg = 1 + (wb_param[1])/wb_bar_max
    kb = 1 + (wb_param[0])/wb_bar_max
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    img = cv2.merge([b, g, r])

    return img


if __name__ == '__main__':
    # demo to show the Camera_Thread class usage
    import cv2

    record_video_flag = True # 允许录制
    start_record_flag = False # 开始录制
    out_num = 1

    # initialize
    cap = Camera_Thread(0, video=False, video_path='yolov5_6/myData/videos/1.mp4')
    # cap2 = Camera_Thread(1)

    # record video
    if record_video_flag:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0
        # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # size = (1280, 1024)
        size = (3088, 2064)
        name = 'record.mp4'
        out = cv2.VideoWriter(name,fourcc, fps, size)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
    name_id = 0

    while True:
        # check whether the camera is working, if not, try to reopen it
        if not cap.is_open():
            cap.open()

        # receive one frame
        flag, frame = cap.read()
        # flag2, frame2 = cap2.read()
        # wb = adjust_wb()
        if whiteBalance:
            frame = white_balance(frame)

        if not flag:
            time.sleep(0.1)
            continue

        # 录制中
        if start_record_flag:
            out.write(frame)

        cv2.imshow("frame", frame)
        # cv2.imshow("frame2", frame2)

        k = cv2.waitKey(1)
        if k & 0xff == ord('q'):
            break
        if k & 0xff == ord('r'):
            # record video
            if record_video_flag:
                start_record_flag = True
                print('record video start')
        if k & 0xff == ord('s'):
            name = 'record' + str(name_id) + '.jpg'
            cv2.imwrite(name,frame)  # 写入图片
            name_id += 1
            print('image saved')
        if k & 0xff == ord('d'):
            if record_video_flag:
                start_record_flag = False
                out.release()
                print('record video saved')

    cap.release()
    cv2.destroyAllWindows()
