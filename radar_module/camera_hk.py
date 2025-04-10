'''
相机类
基于海康官方sdk开发，由于实际使用的是USB相机，因此注释和删去掉了网口相机的部分
主要流程参考GrabImage.py
'''
from re import T
import sys
import os
# print('os.getcwd')
# print(os.getcwd())
sys.path.append(os.getcwd())   # 获取当前工作目录
# print('sys.path')
# print(sys.path)

import cv2
import numpy as np
import yaml
import time
from datetime import datetime
from radar_module.config import camera_match_list,CAMERA_CONFIG_DIR,CAMERA_YAML_PATH,CAMERA_CONFIG_SAVE_DIR,preview_location, wb_bar_max, wb_param, whiteBalance
from hk_sdk.MvCameraControl_class import*
from hk_sdk.MvCameraControl_header import*


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
            win_loc = 0 if camera_type in [0,1] else 1
            # 移动至合适位置
            cv2.moveWindow("preview of {0} press T to adjust".format(camera_type), *preview_location[win_loc])
            cv2.imshow("preview of {0} press T to adjust".format(camera_type), frame)
            key = cv2.waitKey(0)
            cv2.destroyWindow("preview of {0} press T to adjust".format(camera_type))
            # 按其他键则不调参使用默认参数，按t键则进入调参窗口，可调曝光和模拟增益
            # waitkey()返回按下的按键对应的ASCII，&0xff只取后八位
            if key == ord('t') & 0xFF:
                cap.NoautoEx()  # 不自动曝光
                # 4.17换的两个海康相机不再作为高帧率相机使用
                tune_exposure(cap)
            cap.saveParam(date)  # 保存参数，以便在比赛中重启时能够自动读取准备阶段设置的参数
        init_flag = True
    except Exception as e:
        # If cap is not open, hcamera will be -1 and cap.release will do nothing. If camera is open, cap.release will close the camera
        cap.release()
        print("[ERROR] {0}".format(e))
    return init_flag, cap

  
   # 主程序入口
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

    # 4.8 相机外参应该是在pose_save里
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
        # 通过序列号可以人为控制相机打开的顺序，如果直接默认打开就不知道先打开哪个
        # hcamera = -1的解释见上面的open_camera()

        deviceList = MV_CC_DEVICE_INFO_LIST()   # 获取设备列表
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE  # 确定设备类型
    
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        
        # 枚举失败
        if ret != 0:
            self.hCamera = -1
            print ("enum devices fail! ret[0x%x]" % ret)
            return 
        
        # 找不到设备
        if deviceList.nDeviceNum == 0:
            self.hCamera = -1
            print ("find no device!")
            return 
        
        # 序列号列表
        existing_camera_name = []
        
        # 遍历读取到的相机
        for i in range(0, deviceList.nDeviceNum):
            # cast应该是内置函数
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents          
            if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                # print ("\nu3v device: [%d]" % i)
                # strModeName = ""
                # for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                #     if per == 0:
                #         break
                #     strModeName = strModeName + chr(per)
                # print ("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)   # 获取序列号

                existing_camera_name.append(strSerialNumber)  # 添加序列号
                # print ("user serial number: %s" % strSerialNumber)
        
        # match_list需要自己在config里面改
        if not camera_match_list[camera_type] in existing_camera_name:
            # 所求相机不存在
            self.hCamera = -1
            return

        camera_no = existing_camera_name.index(camera_match_list[camera_type])  # 所求相机在枚举列表中编号

        # ch:创建相机实例 | en:Creat Camera Object
        self.hCamera = MvCamera()
        
        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(camera_no)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.hCamera.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print ("create handle fail! ret[0x%x]" % ret)
            self.hCamera = -1
            
        # ch:打开设备 | en:Open device
        ret = self.hCamera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print ("open device fail! ret[0x%x]" % ret)
            self.hCamera = -1
	
        self.camera_type = camera_type
        
        # 设置触发模式为连续采集
        ret = self.hCamera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print ("set trigger mode fail! ret[0x%x]" % ret)
            self.hCamera = -1
            return
        
        # 海康相机似乎没有专门需要自己处理黑白和彩色相机
        
        # 读取配置文件
        # 海康相机的文件配置参考 https://blog.csdn.net/qq_23107577/article/details/114261188
        # 暂时先用Featureload()和featuresave()方法，就在相机类里，和mindvision的处理不同
        if not is_init:
            # default camera parameter config
            param_path = "{0}/camera_{1}.ini".format(CAMERA_CONFIG_DIR,
                                                        camera_type)

            ret = self.hCamera.MV_CC_FeatureLoad(param_path)
            if ret != 0:
                print ("load param failed! ret[0x%x]" % ret)

        else:
            # 初次启动后，保存的参数文件
            param_path = "{0}/camera_{1}_of_{2}.ini".format(CAMERA_CONFIG_SAVE_DIR,
                                                               camera_type, path)

            self.hCamera.MV_CC_FeatureLoad(param_path)
            if ret != 0:
                print ("load param failed! ret[0x%x]" % ret)
        
        # 4.16暂时不明这句的意思
        # mvsdk.CameraSetAeState(self.hCamera, 0)
        
        # 4.16 这些参数保存在cameraOpeartion类中，似乎没有专门返回的函数
        # print(f"[INFO] camera exposure time {mvsdk.CameraGetExposureTime(self.hCamera) / 1000:0.03f}ms")
        # print(f"[INFO] camera gain {mvsdk.CameraGetAnalogGain(self.hCamera):0.03f}")

        # ch:开始取流 | en:Start grab image
        ret = self.hCamera.MV_CC_StartGrabbing()
        if ret != 0:
            print ("start grabbing fail! ret[0x%x]" % ret)
            self.hCamera = -1
            return

        # try:
        #     hThreadHandle = threading.Thread(target=self.work_thread, args=(self.hCamera, None, None))  # 传入相机对象
        #     hThreadHandle.start()
        # except:
        #     print ("error: unable to start thread")

    # 为取图线程定义一个函数
    # 4.16 感觉取图线程是不能停止的，即内存分配不能停止，该函数不能有终止条件
    # 但我感觉也可以上面仅仅start grabbing，在read里面getbuffer，只是我不是那么连续罢了
    # 取图的时候，似乎可以用像素格式解析，比如老代码mindvision是这样搞的
    # 4.16 但是海康取图的时候似乎能直接返回BGR24的图，先看看能不能用，而且这种方法不需要释放buffer
    # 4.16 看不懂这个到底怎么搞的，还是参考getbuffer先
    def read(self, pData=0, nDataSize=0):
        if self.hCamera == -1:
            return
        stOutFrame = MV_FRAME_OUT()  # 图像结构体
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))   # 分配空间
        ret = self.hCamera.MV_CC_GetImageBuffer(stOutFrame, 100)  # 最后一个参数是超出时间
        print(stOutFrame.stFrameInfo.enPixelType)
        # 5.2 昨天测试发现本来输出的就已经是Bayer格式，其实也不用海康自己的像素转换了，当然强制转换一下更保险一点，MVS下看两个相机默认的输出格式都是Bayer BG8
        if None != stOutFrame.pBufAddr and 0 == ret: 
            print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
            MvCamCtrldll.memcpy(byref(pData), stOutFrame.pBufAddr,stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
            data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),dtype=np.uint8)
            data = data.reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, -1)
            if stOutFrame.stFrameInfo.enPixelType != 17301515:     # 5.1 PixelType_Gvsp_BayerBG8 格式
                pass
                # convert_param = MV_CC_PIXEL_CONVERT_PARAM()
                # convert_param.nWidth = stOutFrame.stFrameInfo.nWidth  # 图像宽
                # convert_param.nHeight = stOutFrame.stFrameInfo.nHeight  # 图像高
                # convert_param.pSrcData = stOutFrame.pBufAddr  # 输入数据缓存
                # convert_param.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen  # 输入数据大小
                # convert_param.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType  # 输入像素格式
                # convert_param.enDstPixelType = self.hCamera.PixelType_Gvsp_BGR8_Packed  # 输出像素格式
                # convert_param.pDstBuffer = data  # 输出数据缓存
                # convert_param.nDstBufferSize = size  # 输出缓存大小
                # self.hCamera.MV_CC_ConvertPixelType(convert_param)
            else:
                frame = cv2.cvtColor(data, cv2.COLOR_BAYER_BG2RGB)
            
            # 释放
            ret = self.hCamera.MV_CC_FreeImageBuffer(stOutFrame)
            return True, frame

        else:
            print ("no data[0x%x]" % ret)
            ret = self.hCamera.MV_CC_FreeImageBuffer(stOutFrame)
            return False, None
    
    def setExposureTime(self, ex=30):
        if self.hCamera == -1:
            return
        ret = self.hCamera.MV_CC_SetFloatValue("ExposureTime", ex)
        if MV_OK != ret:   # MV_OK就是0
            print("Set ExposureTime fail! nRet [0x%x]\n", ret)
            return 
        


    def setGain(self, gain):
        if self.hCamera == -1:
            return
        # 海康相机的模拟增益似乎只能到12左右
        ret = self.hCamera.MV_CC_SetFloatValue("Gain", gain)
        if MV_OK != ret:   # MV_OK就是0
            print("Set Gain fail! nRet [0x%x]\n", ret)
            return 
        

    def saveParam(self, path):

        if self.hCamera == -1:
            return
        if not os.path.exists(CAMERA_CONFIG_SAVE_DIR):
            os.mkdir(CAMERA_CONFIG_SAVE_DIR)
        param_path = "{0}/camera_{1}_of_{2}.ini".format(CAMERA_CONFIG_SAVE_DIR, self.camera_type, path)

        # 海康有几种读取和保存参数的手段，这里先用.ini格式试试
        ret = self.hCamera.MV_CC_FeatureSave(param_path)
        if MV_OK != ret:
            print("Save Feature fail! nRet [0x%x]\n", ret);

    def NoautoEx(self):
        '''
        设置不自动曝光
        '''
        if self.hCamera == -1:
            return
        
        # 在开发手册里找的，相机参数节点表
        ret = self.hCamera.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        if ret != 0:
            print ("set NoautoEx fail! ret[0x%x]" % ret)
            # self.hCamera = -1
            return

    def getExposureTime(self):
        if self.hCamera == -1:
            return -1
        # 海康的获取参数流程比较奇怪，这里参考例程Get_Parameter()方法
        stFloatParam_exposureTime = MVCC_FLOATVALUE()
        memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.hCamera.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
        return stFloatParam_exposureTime.fCurValue

    def getAnalogGain(self):
        if self.hCamera == -1:
            return -1
        stFloatParam_gain = MVCC_FLOATVALUE()
        memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.hCamera.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
        return stFloatParam_gain.fCurValue

    def release(self):
        if self.hCamera == -1:
            return
        
        # ch:停止取流 | en:Stop grab image
        ret = self.hCamera.MV_CC_StopGrabbing()
        if ret != 0:
            print ("stop grabbing fail! ret[0x%x]" % ret)
            return
         # ch:关闭设备 | Close device
        ret = self.hCamera.MV_CC_CloseDevice()
        if ret != 0:
            print ("close deivce fail! ret[0x%x]" % ret)
            return    # 引发异常并退出

        # ch:销毁句柄 | Destroy handle
        ret = self.hCamera.MV_CC_DestroyHandle()
        if ret != 0:
            print ("destroy handle fail! ret[0x%x]" % ret)
            return

# 4.17 这部分其实纯纯UI，其他的控制函数改好了这部分功能直接继承
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
        cv2.setTrackbarMax("gain", "exposure press q to exit", 12)
        cv2.setTrackbarMin("gain", "exposure press q to exit", 0)
        # cv2.setTrackbarPos("gain", "exposure press q to exit", int(cap.getAnalogGain()))
        cv2.setTrackbarPos("gain", "exposure press q to exit",8) # 默认增益
    else:
        cv2.createTrackbar("exposure ", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("exposure ", "exposure press q to exit", 100)
        cv2.setTrackbarMin("exposure ", "exposure press q to exit", 0)
        # cv2.setTrackbarPos("exposure ", "exposure press q to exit", int(cap.getExposureTime()))
        cv2.setTrackbarPos("exposure ", "exposure press q to exit", 20) # 默认曝光时间ms
        cv2.createTrackbar("gain", "exposure press q to exit", 0, 1, lambda x: None)
        cv2.setTrackbarMax("gain", "exposure press q to exit", 12)
        cv2.setTrackbarMin("gain", "exposure press q to exit", 0)
        # cv2.setTrackbarPos("gain", "exposure press q to exit", int(cap.getAnalogGain()))
        cv2.setTrackbarPos("gain", "exposure press q to exit", 8) # 默认增益

    # white balance
    # auto param
    if not high_fps: # 云台相机不调节白平衡，感觉是主相机要调白平衡
        flag, frame = cap.read()
        img, b_init, g_init, r_init = auto_white_balance_once(frame) # 自己先计算一个初值出来，然后在UI调节时显示一个
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
        cv2.setTrackbarPos("wb_r", "exposure press q to exit", r_init)     # 5.2 报错可能是因为窗口名称不一致 什么Null window handler 有可能是负数，不知是否有必要改

    flag, frame = cap.read()
   
    # 获取参数值并修改
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
                frame = white_balance(frame, wb_param)   # 这也是自己写的算法

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