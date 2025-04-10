'''
自定义UI类
使用Qt设计的自定义UI
'''
import sys
import cv2
import os
import numpy as np
from datetime import datetime

from radar_module.config import MAP_PATH, INIT_FRAME_PATH, INIT_FRAME_2_PATH, INIT_FRAME_3_PATH, INIT_FRAME_4_PATH, VIDEO_SAVE_DIR, PC_RECORD_FRAMES, reverse, enemy2color, record_fps, unit_list
from radar_module.camera import read_yaml

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap
from qt_design_tools.radar_ui import Ui_MainWindow  # 加载自定义的布局

from radar_module.Lidar import Radar
from radar_module.config import using_dq

from UART import UART_passer

stop_record = False
open_thread = True
 
# QT5 UI
class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):  # 这个地方要注意Ui_MainWindow
    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)  # 初始化ui

        # 4.19 这些图片规定了各个部分的尺寸大小，先改成现在的2600*2160
        frame = cv2.imread(INIT_FRAME_PATH)
        frame_map = cv2.imread(MAP_PATH)
        frame_2 = cv2.imread(INIT_FRAME_2_PATH)
        frame_3 = cv2.imread(INIT_FRAME_3_PATH)
        frame_4 = cv2.imread(INIT_FRAME_4_PATH)

        # 录制保存位置
        self.save_address = VIDEO_SAVE_DIR
        try:
            if not os.path.exists(self.save_address):
                os.mkdir(self.save_address)
        except: # 当出现磁盘未挂载等情况，导致文件夹都无法创建
            print("[ERROR] The video save dir even doesn't exist on this computer!")
        self.save_title = '' # 当场次录制文件夹名，左右相机放一个文件夹下


        # 小地图翻转
        if reverse: # 我方B
            # print('我方B敌方R')
            frame_map = cv2.rotate(frame_map,cv2.ROTATE_90_CLOCKWISE)
        else: # 我方R
            # print('我方R敌方B')
            frame_map = cv2.rotate(frame_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        self.set_image(frame,"main_demo")
        self.set_image(frame_map,"map")
        self.set_image(frame_2,"second_cam")
        self.set_image(frame_3,"third_cam")
        self.set_image(frame_4,"base_img")
        del frame, frame_map, frame_2, frame_3, frame_4
        self.btn1.setText("中间相机标定位姿")
        self.btn2.setText("开始录制")   # 4.18 这部分需修改为两个相机都开启录制
        self.btn3.setText("中间相机ON")  # 4.18
        self.locate_pick_start = False
        self.locate_pick_state = False
        self.record_state = [False, False, False]
        # self.base_view = False
        self.camera0_view = True   # 5.2 默认开启中间相机视角
        self.camera1_view = False
        # self.camera2_view = False
        self.feedback_message_box = [] # feedback信息列表
        self.record_object = [None, None]   # 相机录制对象

    def btn1_clicked(self):
        """
        按钮1功能-----位姿标定
        """
        if self.locate_pick_state:# 标定完成
            self.btn1.setText("再次位姿标定中")
            self.set_text("feedback","再次位姿标定中")
            self.locate_pick_start = True
        else: # 按下，开始标定
            self.btn1.setText("位姿标定中")
            self.set_text("feedback","位姿标定中")
            self.locate_pick_start = True
            
    def btn2_clicked(self):
        '''
        按钮2功能-----录像
        '''
        global stop_record, open_thread
        if self.record_state[0] and self.record_state[1]:
            # 结束录制
            print('按键结束录制')
            self.btn2.setText("开始录制") # 复位
            stop_record = True
            open_thread = False

            # 单相机录制
            # video0 = self.record_object
            # video0.release()
            # 多相机录制:
            # video0 = self.record_object[0]
            # video1 = self.record_object[1]
            # video2 = self.record_object[2]
            # if video0 is not None:
            #     video0.release()
            # if video1 is not None:
            #     video1.release()
            # if video2 is not None:
            #     video1.release()

            if not using_dq:
                Radar.stop_record()

            # self.record_object[0] = None
            # self.record_object[1] = None
            # self.record_object[2] = None
            save_address = os.path.join(self.save_address,self.save_title)
            self.set_text("feedback", "左右相机录制已保存于{0}".format(save_address))

            self.record_state = [False, False]
        else:
            # 开始录制
            print("按键开始录制")
            stop_record = False
            open_thread = True
            # if not os.path.exists(self.save_address):
            #     print("[ERROR] path not existing")
            #     return
            self.btn2.setText("停止录制") # 复位

            # 4.19 录制仅仅靠文件名区分
            # 每次开始都只需要创建一个文件夹即可
            title = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            # os.mkdir(os.path.join(self.save_address,title))
            self.save_title = title
            
            # 中间相机开始录制
            name0 = os.path.join(self.save_address, title, "camera0.mp4")
            # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            # camera_0_size = read_yaml(0)[4]   
            # self.record_object[0] = (cv2.VideoWriter(name0, fourcc, record_fps, camera_0_size))
            self.record_state[0] = True
            
            # 左相机开始录制
            name1 = os.path.join(self.save_address, title, "camera1.mp4")
            # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            # camera_1_size = read_yaml(1)[4]   
            # self.record_object[1] = (cv2.VideoWriter(name1, fourcc, record_fps, camera_1_size))
            self.record_state[1] = True

            # 左相机开始录制(已废弃)
            # name2 = os.path.join(self.save_address, title, "camera2.mp4")
            # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            # camera_2_size = read_yaml(2)[4]  
            # self.record_object[2] = (cv2.VideoWriter(name2, fourcc, record_fps, camera_2_size))
            # self.record_state[2] = True

            if not using_dq:
                Radar.start_record(PC_RECORD_FRAMES)
                
            self.set_text("feedback", "录制已开始于{0}".format(title))
            print('start recording video0')
            print(name0)
            print('start recording video1')
            print(name1)
            # print('start recording video2')
            # print(name2)
    
    # 4.18 将按钮3功能修改为切换相机视角
    def btn3_clicked(self):
        """
        按钮3功能-----相机视角切换
        """
        
        # 5.3 三个相机视角轮流切换
        if self.camera0_view:# 正显示中间相机画面
            self.btn3.setText("左相机ON")   # 此时应该已经按下完成切换到右相机
            self.set_text("feedback","切换至左相机")
            self.camera0_view = False
            self.camera1_view = True
        elif self.camera1_view:  # 正显示右相机画面
            self.btn3.setText("中间相机ON")
            self.set_text("feedback","切换至中间相机")
            self.camera1_view = False
            self.camera0_view = True
        # elif self.camera2_view:  # 正显示右相机画面
        #     self.btn3.setText("中间相机ON")
        #     self.set_text("feedback","切换至中间相机")
        #     self.camera2_view = False
        #     self.camera0_view = True
    
    # 5.2 qt会自动把相机发布的照片自适应UI里面定义的大小
    def set_image(self, frame, position=""):
        """
        Image Show Function

        :param frame: the image to show
        :param position: where to show
        :return: a flag to indicate whether the showing process have succeeded or not
        """
        if not position in ["main_demo", "map","message_box","second_cam","third_cam","base_img"]:
            print("[ERROR] The position isn't a member of this UIwindow")
            return False

        # get the size of the corresponding window
        if position == "main_demo":
            width = self.main_demo.width()
            height = self.main_demo.height()
        elif position == "map":
            width = self.map.width()
            height = self.map.height()
        elif position == "message_box":
            width = self.message_box.width()
            height = self.message_box.height()
        elif position == "second_cam":
            width = self.second_cam.width()
            height = self.second_cam.height()
        elif position == "third_cam":
            width = self.third_cam.width()
            height = self.third_cam.height()
        elif position == "base_img":
            width = self.base_img.width()
            height = self.base_img.height()

        if frame.ndim == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            return False

        # allocate the space of QPixmap
        temp_image = QImage(rgb,rgb.shape[1],rgb.shape[0], QImage.Format_RGB888)
        temp_pixmap = QPixmap(temp_image).scaled(width,height)

        # set the image to the QPixmap location to show the image on the UI
        if position == "main_demo":
            self.main_demo.setPixmap(temp_pixmap)
            self.main_demo.setScaledContents(True)
        elif position == "map":            # 5.2 这个map的尺寸似乎是严格不能变的，不然会变形
            self.map.setPixmap(temp_pixmap)
            self.map.setScaledContents(True)
        elif position == "message_box":
            self.message_box.setPixmap(temp_pixmap)
            self.message_box.setScaledContents(True)
        elif position == "second_cam":
            self.second_cam.setPixmap(temp_pixmap)
            self.second_cam.setScaledContents(True)
        elif position == "third_cam":
            self.third_cam.setPixmap(temp_pixmap)
            self.third_cam.setScaledContents(True)
        elif position == "base_img":
            self.base_img.setPixmap(temp_pixmap)
            self.base_img.setScaledContents(True)
        return True

    def set_text(self, position: str, message=""):
        """
        to set text in the QtLabel

        :param position: must be one of the followings: "feedback", "message_box", "state"
        :param message: For feedback, a string you want to show in the next line;
        For the others, a string to show on that position , which will replace the origin one.
        :return:
        a flag to indicate whether the showing process have succeeded or not
        """
        if position not in ["feedback", "message_box"]:
            print("[ERROR] The position isn't a member of this UIwindow")
            return False

        if position == "feedback":
            if len(self.feedback_message_box) >= 5: # the feedback could contain at most 12 messages lines.
                self.feedback_message_box.pop(0)
            self.feedback_message_box.append(message)
            # Using "<br \>" to combine the contents of the message list to a single string message
            message = "<br \>".join(self.feedback_message_box) # css format to replace \n
            self.feedback.setText(message)
            return True
        if position == "message_box":
            self.message_box.setText(message)
            return True

# UI HP与比赛状态显示
class HP_scene(object):
    _stage_max = 5 # 5格血量条
    _size = (480, 310)
    _font_size = 0.8
    _outpost = 1500 # 前哨站血量上限
    _base = 5000 # 基地血量上限
    _sentry = 1000 # 哨兵血量上限
    _font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, reverse, show_api):
        '''
        展示一个血量条

        :param reverse:reverse r b
        :param show_api:在主UI显示调用的api f(img:np.ndarray)
        '''
        self._scene = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
        self._show_api = show_api
        self._enermy = 1 if reverse==0 else 0 # 与上交相反 reverse = 0我方R:_enemy = 1
        blue_color = (255, 0 ,0)
        red_color = (0, 0, 255)
        self._puttext = lambda txt, x, y, color: cv2.putText(self._scene, txt, (x, y), self._font,
                                                      self._font_size, color, 2)

        # 标题
        if self._enermy:
            self._puttext("OUR {0}".format(enemy2color[not self._enermy]), 10, 25, red_color)
            self._puttext("ENEMY {0}".format(enemy2color[self._enermy]), 250, 25, blue_color)
        else:
            self._puttext("OUR {0}".format(enemy2color[not self._enermy]), 10, 25, blue_color)
            self._puttext("ENEMY {0}".format(enemy2color[self._enermy]), 250, 25, red_color)
        # 单位名称
        if self._enermy:
            # OUR CAR
            for i in range(8):
                self._puttext("{0}".format(unit_list[i + 8 * (not self._enermy)]), 10, 60 + 30 * i, red_color)
            # enemy
            for i in range(8):
                self._puttext("{0}".format(unit_list[i + 8 * (self._enermy)]), 250, 60 + 30 * i, blue_color)
        else:
            # OUR CAR
            for i in range(8):
                self._puttext("{0}".format(unit_list[i + 8 * (not self._enermy)]), 10, 60 + 30 * i, blue_color)
            # enemy
            for i in range(8):
                self._puttext("{0}".format(unit_list[i + 8 * (self._enermy)]), 250, 60 + 30 * i, red_color)

        # 划分线
        cv2.line(self._scene, (0, 32), (self._size[0], 32), (255, 255, 255), 2)
        cv2.line(self._scene, (240, 0), (240, 275), (255, 255, 255), 2)
        cv2.line(self._scene, (0, 185), (self._size[0], 185), (255, 255, 255), 2)
        cv2.line(self._scene, (0, 275), (self._size[0], 275), (255, 255, 255), 2)
        self._out_scene = self._scene.copy()

    # 绘制1个HP条
    def _put_hp(self,hp , hp_max, x, y):
        # 血量条长度MAX 100pixel
        hp_m = int(hp_max)
        if not hp_m:
            hp_m = 100
        radio = hp/hp_m
        if radio > 0.6:
            color = (20,200,20) # 60%以上绿色
        elif radio > 0.2:
            color = (0,255,255) # 20%以上黄色
        else:
            color = (0,0,255) # 20%以下红色
        # 画血量条
        width = 100 //self._stage_max
        cv2.rectangle(self._out_scene,(x,y),(x+int(100*radio),y+15),color,-1)
        # 画血量格子
        for i in range(self._stage_max):
            cv2.rectangle(self._out_scene,(x+i*width,y),(x+(i+1)*width,y+15),(255,255,255),2)

    def update(self, HP, max_hp):
        '''
        根据读取到的血量和计算的血量上限，绘制血量信息
        '''
        # our
        for i in range(8):
            if i < 5: # 1-5
                hp = HP[i + 8 * (not self._enermy)]
                self._put_hp(hp, max_hp[i + 5 * (not self._enermy)], 60, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)
            if i == 5: # sentry
                hp = HP[i + 8 * (not self._enermy)]
                self._put_hp(hp,self._sentry, 60, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)
            if i == 6:
                hp = HP[i + 8 * (not self._enermy)]
                self._put_hp(hp, self._outpost, 60, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)
            if i == 7:
                hp = HP[i + 8 * (not self._enermy)]
                self._put_hp(hp, self._base, 60, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)

        # enemy
        for i in range(8):
            if i < 5 : # 1-5
                hp = HP[i + 8 * (self._enermy)]
                self._put_hp(hp, max_hp[i+5*self._enermy], 60+240, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170+240, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)
            if i == 5: # sentry
                hp = HP[i + 8 * (self._enermy)]
                self._put_hp(hp, self._sentry, 60+240, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170+240, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)
            if i == 6:
                hp = HP[i + 8 * (self._enermy)]
                self._put_hp(hp, self._outpost, 60+240, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170+240, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)
            if i == 7:
                hp = HP[i + 8 * (self._enermy)]
                self._put_hp(hp, self._base, 60+240, 42 + 30*i)
                cv2.putText(self._out_scene, "{0}".format(hp), (170+240, 56 + 30*i), self._font,
                            0.6, (255, 255, 255), 2)

    def update_stage(self,stage,remain_time,BO,BO_max):
        '''
        显示比赛阶段和BO数
        '''
        cv2.putText(self._out_scene,"{0} {1}s      BO:{2}/{3}".format(stage,remain_time,BO,BO_max),(20,300),self._font,
                                                      0.6, (255, 255, 255), 2)
    def show(self):
        '''
        和其他绘制类一样，显示
        '''
        self._show_api(self._out_scene)

    def refresh(self):
        '''
        和其他绘制类一样，换原始未绘制的画布，刷新
        '''
        self._out_scene = self._scene.copy()


if __name__ == '__main__':
    # demo of the window class
    app = QApplication(sys.argv)
    QTwindow = Mywindow()

    # QTwindow.set_text("message_box",'<br \>'.join(['',"<font color='#FF0000'><b>base detect enermy</b></font>","<font color='#FF0000'><b>base detect enermy</b></font>",
    #                               f"哨兵:<font color='#FF0000'><b>{99:d}</b></font>"]))

    QTwindow.show()

    SELF_COLOR = ['RED', 'BLUE']
    QTwindow.set_text("message_box", f"You are {SELF_COLOR[reverse]:s}") # 初始化显示，enemy是否选对
    hp_scene = HP_scene(reverse, lambda x: QTwindow.set_image(x, "message_box"))

    frame_m = cv2.imread(MAP_PATH)
    cap = cv2.VideoCapture(0)


    while True:
        # 需要循环show一个opencv窗口才能保证QtUI的更新(任意图片均可)
        cv2.imshow("out",frame_m)

        ret, frame = cap.read() #逐帧读取影片

        # 显示hp信息
        UART_passer.get_message(hp_scene)
        hp_scene.show()  # update the message box
        
        QTwindow.set_image(frame, "main_demo")
        # cv2.imshow("out",frame)
        QTwindow.set_image(frame, "map")

        key = cv2.waitKey(1)

        pass

    sys.exit(app.exec_())