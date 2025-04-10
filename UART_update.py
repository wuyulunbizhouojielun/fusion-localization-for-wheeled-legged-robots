'''
裁判系统统合类
'''
import time
import numpy as np
from serial_package import offical_Judge_Handler, Game_data_define
import queue

from radar_module.config import reverse, BO, ARER_LENGTH, AREA_WIDTH, home_test, debug_info
# enemy = 1 if reverse==0 else 0 # 与上交相反
###### 采自官方demo ###########
ind = 0 # 每次执行死循环发送id序号（0-5）用以控制发送频率
ind_our_coor = 0
Id_red = 1
Id_blue = 101

buffercnt = 0    # 可能表示的是数据帧长度
buffer = [0]
buffer *= 1000
cmdID = 0
indecode = 0

# 今年加入新的规则，可以添加新的数据头
Game_state = Game_data_define.game_state()
Game_result = Game_data_define.game_result()
Game_robot_HP = Game_data_define.game_robot_HP()
# Game_dart_status = Game_data_define.dart_status()
Game_event_data = Game_data_define.event_data()
Game_supply_projectile_action = Game_data_define.supply_projectile_action()
Game_refree_warning = Game_data_define.refree_warning()
Game_dart_info = Game_data_define.dart_info_t()
Game_radar_info = Game_data_define.radar_info_t()
Game_radar_cmd = Game_data_define.radar_cmd_t()
Game_mark_data = Game_data_define.mark_data_t()

# 每次收到的数据帧大致可以分为帧头+数据+校验位+帧尾
# 总体来看，就是一个接收裁判系统数据，以及给裁判系统或者己方机器人发送数据
# 上交的开源毕竟是21年的，还是先不参考这部分了

# main.py传进来的包括己方、敌方车辆坐标信息、预警信息、血量UI，其他的在本模块内实现即可

# 串口通信接收线程
def read(ser):
    global buffercnt
    buffercnt = 0
    global buffer
    global cmdID
    global indecode
    # TODO:qt thread

    while True:
        s = ser.read(1)   # 读取数据，默认一字节
        s = int().from_bytes(s, 'big')  # byte转化为整数，'big'左边高位右边低位

        if buffercnt > 50:
            buffercnt = 0

        # print(buffercnt)
        # 5.7 buffer是根据串口读取到的信息，用buffercount在对应位置单字节单字节更新
        buffer[buffercnt] = s
        # print(hex(buffer[buffercnt]))

        if buffercnt == 0:
            if buffer[buffercnt] != 0xa5:   # 0xa5是数据帧起始字节，估计是每种数据帧都是以此开头
                buffercnt = 0
                continue

        if buffercnt == 5:
            if offical_Judge_Handler.myVerify_CRC8_Check_Sum(id(buffer), 5) == 0:   # id()获取内存地址，帧头CRC8校验第五位，校验失败应该是返回0
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 7:
            # 位操作，应该是低位先行，命令码都是16位，一位16进制相当于4为二进制，高位需要向左移位
            cmdID = (0x0000 | buffer[5]) | (buffer[6] << 8)   # 数据头格式，第6、7位是cmd 
            # print("cmdID = {}" .format(cmdID))
        # if cmdID == 0x0303:
        #     print('0x%x'%cmdID)

        # 比赛状态数据，1Hz（比赛阶段、剩余时间等）
        # data 11byte，帧头帧尾命令码9byte
        # 这下应该就很清晰了，buffercnt获取数据总长度，并用cmdID获取信息类型，对帧尾进行CRC16校验
        if buffercnt == 20 and cmdID == 0x0001:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 20):
                UART_passer.Referee_Update_GameData()
                buffercnt = 0   # 收完数据后归零
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        # 比赛结果数据，比赛结束后发送
        if buffercnt == 10 and cmdID == 0x0002:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Referee_Game_Result()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        # 各车血量
        # 5.7 感觉没什么用，暂时注释掉不处理
        if buffercnt == 41 and cmdID == 0x0003:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 41):
                # UART_passer.Referee_Robot_HP()  
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        # if buffercnt == 12 and cmdID == 0x0004:
        #     if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 12):
        #         Referee_dart_status()
        #         buffercnt = 0
        #         if buffer[buffercnt] == 0xa5:
        #             buffercnt = 1
        #         continue

        # 场地事件数据，1Hz（机关、增益点状态等）
        if buffercnt == 13 and cmdID == 0x0101:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Referee_event_data()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        # 补给站动作标识数据，1Hz（动作改变后发送）
        # 4.23 长度似乎应改为13，原来是12
        if buffercnt == 13 and cmdID == 0x0102:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Refree_supply_projectile_action()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

         # 裁判警告数据
         # 4.23 裁判警告数据改为3byte
        if buffercnt == 12 and cmdID == 0x0104:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 12):
                Refree_Warning()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

         # 飞镖发射口倒计时
         # 4.23 修改为飞镖发射相关数据，数据段变为3byte，结构体也要改
        if buffercnt == 12 and cmdID == 0x0105:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 12):
                Refree_dart_remaining_time()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue
        
        # 4.24 这部分雷达都没有
        
        # 4.24 大小修改
        if buffercnt == 22 and cmdID == 0x201:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 27):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 25 and cmdID == 0x202:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 25):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 25 and cmdID == 0x203:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 25):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue
        
        # 4.24 增益数据大小修改
        if buffercnt == 15 and cmdID == 0x204:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 10 and cmdID == 0x206:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 13 and cmdID == 0x209:  # 雷达没有
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        # 机器人间自定义交互数据，上限10Hz
        # 5.7 没做任何处理，仅仅终端打印接收
        if buffercnt == 17 and cmdID == 0x301:  # 2byte数据
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 17):
                # 比赛阶段信息
                UART_passer.Receive_Robot_Data()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue
        
        # 小地图
        if buffercnt == 24 and cmdID == 0x0303:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 24):
                # 云台手通信
                Refree_Aerial_Message()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue
        
        # 雷达自主决策
        if buffercnt == 10 and cmdID == 0x020E:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Receive_radar_info()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue
        
        # 标记进度数据
         if buffercnt == 15 and cmdID == 0x020C:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 15):
                Mark_data()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        buffercnt += 1


# 串口通信发送线程
# 3.11 为什么被注释掉了？
# 4.24 主函数调用write函数，在此执行一直写入，所以通信部分都得放在这里
def write(ser):
    
    while True:
        # 01 小地图坐标发送任务
        Robot_Data_Transmit_Map(ser)
        
        # 车间通信 给哨兵发送坐标
        Our_Robot_Coor_Transmit(ser)

        # 车间通信，发给飞机
        # 5.13 发给云台手让他来触发易伤不太现实，不过他可以根据这个信息判断局势，因为标记进度>100是易伤的充分条件
        Transmit_with_plane(ser)
        
        # 有易伤资格就触发易伤
        if UART_passer.radar_vulnerable_chances > 0 and UART_passer.enemy_being_vulnerable == 0:
            radar_trigger_double_vulnerable()

        # 02 车间通信：预警发送任务
        # flag, code, send_target, alarm_targets, team = UART_passer.pop() # code为预警区域唯一编号==（优先级）0~n
        # if flag:
        #     for t in send_target:
        #         if reverse: # 我方车间通信目标转换为（蓝方）
        #             t += 100
        #         dataID = 0x0200 + (code & 0xFF) # 自定义内容，0x200-0x2ff

        #         # if code == 0: # 反导
        #             # send_t = UART_passer.Remain_time
        #         #     Referee_Transmit_BetweenCar(dataID, t, [send_t&0xff, (send_t&0xff00)>>8, 0, 0], ser) # 包数据端，发送反导信息时，比赛剩余时间
        #         target_code = 0
        #         for target in alarm_targets: # 进入预警区域的敌方单位编号
        #             if debug_info:
        #                 print('target = {}' .format(target))
        #             if target <= 5:
        #                 target -= 1
        #                 target_code += 1<<(target%5) # 基于二进制编码，从右到左各位若为1则表示该位置车辆在预警区域内（从1开始数）
        #             data = [target_code, 0, 0, 0]
        #             # print('dataID = {0} target = {1} data = {2}' .format(dataID, t, data)) # dataID = 513 target = 5 data = [2, 0, 0, 0]
        #             Referee_Transmit_BetweenCar(dataID, t, data, ser)

        #         # if code in [3,4]: # 对于两个高地坡，需要发送该坡为己方坡还是敌方坡 team 1 is enemy, 0 is ours
        #         #     Referee_Transmit_BetweenCar(dataID, t, [target_code, (team==reverse)&0xFF, 0, 0], ser) # 己方坡
        #         # else:
        
        # 03 车间通信：己方坐标发送任务
        # Our_Robot_Coor_Transmit(ser)

        # else:
            # time.sleep(0.1) # 如果仅有预警任务执行（或其他任务中没有sleep），需要加入sleep防止阻塞主线程



################################################

class UART_passer(object):
    '''
    convert the Judge System message
    自定义裁判系统统合类
    '''
    # 4.24 这里存储了从裁判系统接收以及要发出的变量

    # message box controller
    _bytes2int = lambda x:(0x0000 | x[0]) | (x[1] << 8)
    _hp_up = np.array([100,150,200,250,300,350,400,450,500]) # 各个升级血量阶段
    _init_hp = np.ones(10,dtype = int)*500 # 初始化所有单位血量500
    _last_hp = _init_hp.copy()
    _HP = np.ones(16, dtype = int)*500 # to score the received hp message, initialize all as 500
    _max_hp = _init_hp.copy() # 初始化所有单位血量上限500
    _set_max_flag = False # 比赛开始时将单位血量设置为上限

    # location controller
    # 3.11 哨兵不算吗？
    # 4.24 加入哨兵
    _robot_location = np.zeros((6,2),dtype=np.float32) # 敌方单位坐标
    _our_robot_location = np.zeros((6,2),dtype=np.float32) # 我方单位坐标

    # alarming event queue with priority
    _queue = queue.PriorityQueue(-1)
    _queue_our_coor = queue.PriorityQueue(-1)
    # Game State controller
    _BO = 0
    _stage = ["NOT START", "PREPARING", "CHECKING", "5S", "PLAYING", "END"]
    _Now_Stage = 0
    _Game_Start_Flag = False
    _Game_End_Flag = False
    Remain_time = 0
    _HP_thres = 0 # 发送预警的血量阈值，即预警区域中血量最高的车辆的血量低于该阈值则不发送预警
    _prevent_time = [2.,2.,2.,2.,2.,2.] # sleep time 预警发送沉默时间，若距离上次发送小于该阈值则不发送（2S）
    _event_prevent = np.zeros(6) # 距离时间记录
    loop_send = 0 # 在一次小地图车辆id循环中发送的次数
    loop_send_our_coor = 0
    # Aerial control flag 云台手控制置位符
    aerial_record_start = False
    aerial_record_stop = False
    aerial_change_view = False
    # 雷达获得的易伤次数及对方是否正在被触发双倍易伤
    radar_vulnerable_chances = 0
    enemy_being_vulnerable = 0

    @staticmethod
    def _judge_max_hp(HP):
        '''
        血量最大值变化判断逻辑，by 何若坤，只适用于21赛季规则
        '''
        # 3.11 需要比较这赛季和21赛季的血量计算规则
        
        # 5.7 暂时没用
        mask_zero = UART_passer._last_hp > 0 # 血量为0不判断
        focus_hp = HP[[0,1,2,3,4,8,9,10,11,12]] # 只关心这些位置的血量上限  ？？
        # 工程车血量上限不变，不判断
        mask_engineer = np.array([True]*10)
        mask_engineer[[1,6]] = False
        mask_engineer = np.logical_and(mask_zero,mask_engineer)
        # 若血量增加在30到80，则为一级上升（50）
        mask_level1 = np.logical_and(focus_hp-UART_passer._last_hp>30,focus_hp-UART_passer._last_hp<=80)
        # 若血量增加在80以上，则为二级上升（100）
        mask_level2 = focus_hp-UART_passer._last_hp > 80
        UART_passer._max_hp[np.logical_and(mask_level1,mask_engineer)] += 50
        UART_passer._max_hp[np.logical_and(mask_level2,mask_engineer)] += 100
        # 如果有一次上限改变没检测到，使得当前血量大于上限，则调整至相应的上限
        mask_still = np.logical_and(focus_hp > UART_passer._max_hp,mask_engineer)
        for i in np.where(mask_still)[0]:
            UART_passer._max_hp[i] = np.min(UART_passer._hp_up[UART_passer._hp_up > focus_hp[i]])

        UART_passer._last_hp = focus_hp.copy()

    def __init__(self):
        pass

    @staticmethod
    def push_loc(location):
        '''
        放入当前帧位置
        '''
        UART_passer._robot_location = np.float32(location)  # 转化为numpy数组

    @staticmethod
    def get_position():
        '''
        得到当前帧位置
        '''
        return UART_passer._robot_location

    @staticmethod
    def push_our_loc(location):
        '''
        放入当前帧位置
        '''
        UART_passer._our_robot_location = np.float32(location)

    @staticmethod
    def get_our_position():
        '''
        得到当前帧位置
        '''
        return UART_passer._our_robot_location

    @staticmethod
    def _send_check(priority, alarm_target):
        '''
        预警发送判断逻辑

        :param priority:发送优先级
        :param alarm_target:落在预警区域的车辆列表(0-9)

        :return: True if the alarming could be sent
        '''
        if priority < 1: # 飞镖预警不需要判断血量
            return True
        else:
            HP = UART_passer._HP[[0,1,2,3,4,8,9,10,11,12]]
            alarm_tag = np.array(alarm_target,dtype = int)
            alarm_tag = alarm_tag[alarm_tag < 10].tolist()  # 应该是把探测到的车辆序号转成列表
            if (HP[alarm_tag] > UART_passer._HP_thres).any():   # 如果有符合血量阈值的车辆的话
                return True
            else:
                # If all the robot in the alarming region are death, then don't send the alarming.
                return False

    @staticmethod
    def push(priority, send_target, alarm_target, team):
        '''
        将一个预警push进预警队列，以code作为优先级，code越小，优先级越高
        '''
        if UART_passer._send_check(priority,alarm_target):
            UART_passer._queue.put((priority, send_target, alarm_target, team))

    @staticmethod
    def pop():
        '''
        pop出一个预警信息
        '''
        if not UART_passer._queue.empty():
            code, send_targets, alarm_targets, team = UART_passer._queue.get()
            t = time.time()
            # 判断预警间隔
            if t - UART_passer._event_prevent[code] > UART_passer._prevent_time[code]:
                UART_passer._event_prevent[code] = t # reset
                return True, code, send_targets, alarm_targets, team
            else:
                return False,None,None,None,None
        else:
            return False,None,None,None,None

    @staticmethod
    def get_message(hp_scene):
        '''
        基于裁判系统类保存的血量及比赛阶段信息，更新hp信息框
        '''
        pass
        # hp_scene.refresh()
        # hp_scene.update(UART_passer._HP, UART_passer._max_hp)
        # hp_scene.update_stage(UART_passer._stage[UART_passer._Now_Stage],UART_passer.Remain_time,UART_passer._BO+1,BO)

    @staticmethod
    def Referee_Update_GameData():
        # If the game state is from starting to ending or from ending to starting, set stage change flag
        # From Stage "preparing" to "15s" or "5s" or "playing", any of them will enable this(GAME START)

        # 4.24 buffer每个元素一个字节，前面7byte数据头和cmdID，buffer[7]对应比赛状态，右移四位取高四位
        # 5.7 Now_stage 对应6个比赛阶段，两个if来更新比赛状态
        if UART_passer._Now_Stage < 2 and ((buffer[7] >> 4) == 2 or (buffer[7] >> 4) == 3 or (buffer[7] >> 4) == 4):
            # ONE GAME START
            UART_passer._Game_Start_Flag = True
            UART_passer._set_max_flag = True
        if UART_passer._Now_Stage < 5 and (buffer[7] >> 4) == 5:
            # ONE GAME END
            UART_passer._Game_End_Flag = True
            UART_passer._max_hp = UART_passer._init_hp.copy()
        UART_passer._Now_Stage = buffer[7] >> 4
        
        # 这种或和移位的方法来凑2byte
        UART_passer.Remain_time = (0x0000 | buffer[8]) | (buffer[9] << 8)

    @staticmethod
    def Referee_Robot_HP():
        # 1 2 3 4 5 sentry outpost base 血量顺序
        UART_passer._HP = np.array([UART_passer._bytes2int((buffer[i*2-1],buffer[i*2])) for i in range(4,20)],dtype = int)
        if UART_passer._set_max_flag:
            # 比赛开始时，根据读取血量设置最大血量
            UART_passer._max_hp = UART_passer._HP[[0,1,2,3,4,8,9,10,11,12]]
            UART_passer._set_max_flag = False
        else:
            UART_passer._judge_max_hp(UART_passer._HP)

    @staticmethod
    def One_compete_end():
        # check the state change flag
        # 4.24 比赛都结束了，复位
        if UART_passer._Game_End_Flag:
            # reset
            UART_passer._Game_End_Flag = False
            UART_passer._BO += 1
            return True,UART_passer._BO - BO   # BO在config里，需根据比赛场数更改
        else:
            return False,-1

    @staticmethod
    def One_compete_start():
        # check the state change flag
        if UART_passer._Game_Start_Flag:
            # reset
            UART_passer._Game_Start_Flag = False
            return True
        else:
            return False
    
    # 4.24 这部分用来车间通信，目前没做开发
    @staticmethod
    def Receive_Robot_Data():
        '''
        between car receiver api, now no use
        '''
        if (0x0000 | buffer[7]) | (buffer[8] << 8) == 0x0200:
            print("received")


# 4.24 接收雷达自主决策
# 应该低位的数据read到的也是在低位
def Receive_radar_info():
    UART_passer.radar_vulnerable_chances = 2*(buffer[7] & 0x02) + (buffer[7] & 0x01) # bit 1 和 bit 0
    UART_passer.enemy_being_vulnerable = buffer[7] & 0x04  # bit 2

# 5.11 接收各个机器人标记进度数据
def Referee_Mark_data():
    Game_mark_data.mark_hero_progress = buffer[7]
    Game_mark_data.mark_engineer_progress = buffer[8]
    Game_mark_data.mark_standard_3_progress = buffer[9]
    Game_mark_data.mark_standard_4_progress = buffer[10]
    Game_mark_data.mark_standard_5_progress = buffer[11]
    Game_mark_data.mark_sentry_progress = buffer[12]

# 5.11 和无人机之间通信发送标记进度
# 5.13 自定义的data里面datalength包括了我们的发送者接收者，也就是非datalength的只有帧头
def Transmit_with_plane(ser):
    code = 32   # 注意冲突
    dataID = 0x0200 + (code & 0xFF)
    datalength = 12
    if not reverse:
        ReceiverId = 6  # 空中机器人
    else:
        ReceiverId = 106


    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = datalength  # 数据帧中 data 的长度,占两个字节
    buffer[2] = 0
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = 0x01
    buffer[6] = 0x03
    # 自定义内容ID(0x0200-0x02ff之间)
    buffer[7] = dataID & 0x00ff
    buffer[8] = (dataID & 0xff00) >> 8
    # 发自雷达站
    if not reverse:
        buffer[9] = 9
    else:
        buffer[9] = 109
    buffer[10] = 0
    buffer[11] = ReceiverId
    buffer[12] = 0
    # 自定义内容数据段
    buffer[13] = bytes(UART_passer.mark_hero_progress)
    buffer[14] = bytes(UART_passer.mark_engineer_progress)
    buffer[15] = bytes(UART_passer.mark_standard_3_progress)
    buffer[16] = bytes(UART_passer.mark_standard_4_progress)
    buffer[17] = bytes(UART_passer.mark_standard_5_progress)
    buffer[18] = bytes(UART_passer.mark_sentry_progress)

    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), datalength + 9)  # 整包校验

    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + datalength

    for i in range(9 + datalength):
        buffer_tmp_array[i] = buffer[i]
    # if debug_info:
        # print('between car alarm:{}' .format(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))


def Referee_Aerial_Message():
    '''
    Arial controller
    radar keyboard
    云台手车间通信按下按键信息接收
    to be continued...
    '''
    key = buffer[19]
    # if key == ord('R')&0xFF: # 开始录制
    #     UART_passer.aerial_record_start = True
    #     UART_passer.aerial_record_stop = False
    #     print('Refree_Aerial_Message - press - R')
    # if key == ord('D')&0xFF: # 结束录制
    #     UART_passer.aerial_record_stop = True
    #     UART_passer.aerial_record_start = False
    #     print('Refree_Aerial_Message - press - D')
    # if key == ord("C")&0xFF: # 基地视角切换触发信号
    #     UART_passer.aerial_change_view = True
    #     print('Refree_Aerial_Message - press - C')

############# 官方demo的函数 ###########

def Referee_Game_Result():
    # print("Referee_Game_Result")
    Game_result.winner = buffer[7]
    # print("The winner is {0}".format(Game_result.winner))

# def Referee_dart_status():
#     # print("Referee_dart_status")
#     Game_dart_status.dart_belong = buffer[7]
#     Game_dart_status.stage_remaining_time = [buffer[8], buffer[9]]


def Referee_event_data():
    # print("Referee_event_data")
    Game_event_data.event_type = [buffer[7], buffer[8], buffer[9], buffer[10]]

    # doc.write('Referee_event_data' + '\n')

# 4.24 修改为reserved
def Referee_supply_projectile_action():
    # print("Refree_supply_projectile_action")
    Game_supply_projectile_action.reserved = buffer[7]
    Game_supply_projectile_action.supply_robot_id = buffer[8]
    Game_supply_projectile_action.supply_projectile_step = buffer[9]
    Game_supply_projectile_action.supply_projectile_num = buffer[10]

# 4.24 新增count
def Referee_Warning():
    # print("Refree_Warning")
    Game_refree_warning.level = buffer[7]
    Game_refree_warning.foul_robot_id = buffer[8]
    Game_refree_warning.count = buffer[9]

# 4.24 现在变成了dart_info 但是不太明白为什么之前取buffer[8]
# def Refree_dart_remaining_time():
    # print("Refree_dart_remaining_time")
    # Game_dart_remaining_time.time = buffer[8]
    # doc.write('Refree_dart_remaining_time' + '\n')

# 4.24
def Referee_dart_info():
    Game_dart_info.dart_remaining_time = buffer[7]
    Game_dart_info.dart_info = [buffer[8], buffer[9]]  # 打击目标和保留位

####################################

# 4.24 下面两个函数负责车间通信，仿照官方通信帧格式自己定义数据
        
def Referee_Transmit_BetweenCar(dataID, ReceiverId, data, ser):
    '''
    雷达站发送车间通信包函数
    上限10Hz
    '''
    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = 10  # 数据帧中 data 的长度,占两个字节
    buffer[2] = 0
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = 0x01
    buffer[6] = 0x03
    # 自定义内容ID(0x0200-0x02ff之间)
    buffer[7] = dataID & 0x00ff
    buffer[8] = (dataID & 0xff00) >> 8
    # 发自雷达站
    if not reverse:
        buffer[9] = 9
    else:
        buffer[9] = 109
    buffer[10] = 0
    buffer[11] = ReceiverId
    buffer[12] = 0
    # 自定义内容数据段
    buffer[13] = data[0]
    buffer[14] = data[1]
    buffer[15] = data[2]
    buffer[16] = data[3]

    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), 10 + 9)  # 整包校验

    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + 10

    for i in range(9 + 10):
        buffer_tmp_array[i] = buffer[i]
    # if debug_info:
        # print('between car alarm:{}' .format(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))

# 参考官方文档里机器人之间通信的数据体格式
def Referee_Transmit_BetweenCar_Coor(dataID, ReceiverId, datalength, x, y, enemy_x, enemy_y, ser):
    '''
    雷达站发送车间通信包函数
    上限10Hz
    '''
    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = (datalength) & 0x00ff  # 数据帧中 data 的长度,占两个字节
    buffer[2] = ((datalength) & 0xff00) >> 8
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = 0x01
    buffer[6] = 0x03
    # 自定义内容ID(0x0200-0x02ff之间)
    buffer[7] = dataID & 0x00ff
    buffer[8] = (dataID & 0xff00) >> 8
    # 发自雷达站
    if not reverse:
        buffer[9] = 9   # 本方作为雷达站
    else:
        buffer[9] = 109
    buffer[10] = 0
    buffer[11] = ReceiverId
    buffer[12] = 0
    # 自定义内容数据段 己方xy
    buffer[13] = bytes(x)[0]
    buffer[14] = bytes(x)[1]
    buffer[15] = bytes(x)[2]
    buffer[16] = bytes(x)[3]
    buffer[17] = bytes(y)[0]
    buffer[18] = bytes(y)[1]
    buffer[19] = bytes(y)[2]
    buffer[20] = bytes(y)[3]
    # 敌方xy
    buffer[21] = bytes(enemy_x)[0]
    buffer[22] = bytes(enemy_x)[1]
    buffer[23] = bytes(enemy_x)[2]
    buffer[24] = bytes(enemy_x)[3]
    buffer[25] = bytes(enemy_y)[0]
    buffer[26] = bytes(enemy_y)[1]
    buffer[27] = bytes(enemy_y)[2]
    buffer[28] = bytes(enemy_y)[3]

    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), datalength + 9)  # 整包校验

    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + datalength

    for i in range(9 + datalength):
        buffer_tmp_array[i] = buffer[i]
    # if debug_info:
        # print('between car xy:{}' .format(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))

# 本函数向本方所有选手端发送对方坐标，在小地图上显示
def Referee_Transmit_Map(cmdID, datalength, targetId, x, y, ser):
    '''
    小地图信息打包
    上限10Hz

    x，y采用np.float32转换为float32格式
    '''
    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = (datalength) & 0x00ff  # 数据帧中 data 的长度,占两个字节
    buffer[2] = ((datalength) & 0xff00) >> 8
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = cmdID & 0x00ff
    buffer[6] = (cmdID & 0xff00) >> 8
    
    # 4.24 这里是数据段长度，14byte，虽然本来应该是不用的，但上交说协议Bug
    # 坐标为float
    buffer[7] = targetId
    buffer[8] = 0
    buffer[9] = bytes(x)[0]
    buffer[10] = bytes(x)[1]
    buffer[11] = bytes(x)[2]
    buffer[12] = bytes(x)[3]
    buffer[13] = bytes(y)[0]
    buffer[14] = bytes(y)[1]
    buffer[15] = bytes(y)[2]
    buffer[16] = bytes(y)[3]
    buffer[17:20] = [0] * 4 # 朝向，直接赋0，协议bug，不加这一项无效

    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), datalength + 9)  # 等价的
    
    # 4.23 这部分看起来像复制了一份内容
    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + datalength

    for i in range(9 + datalength):
        buffer_tmp_array[i] = buffer[i]
    # if debug_info:
    #     print('tiny map:{}' .format(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))


# 新加入哨兵机器人

def ControlLoop_red():
    '''
    循环红色车辆id
    '''
    global Id_red
    if Id_red == 7:
        Id_red = 1
    else:
        if Id_red == 5:
            Id_red = Id_red + 2  # 跳过空中机器人
        else:
            Id_red = Id_red + 1

def ControlLoop_blue():
    '''
    循环蓝色车辆id
    '''
    global Id_blue
    if Id_blue == 107:
        Id_blue = 101
    else:
        if Id_blue == 105:
            Id_blue = Id_blue + 2
        else:
            Id_blue = Id_blue + 1

# 发送小地图坐标
def Robot_Data_Transmit_Map(ser):
    '''
    向客户端发送车辆坐标以绘制
    放在死循环中执行，上限10Hz
    '''

    # 4.26 ind 和 Id同步循环，一个控制次数一个确定发送的是哪辆车的坐标，目前为循环发送，每次发一辆
    global ind
    location = UART_passer.get_position()[ind] # 得到当前帧位置（来源于主循环中检测出的敌方单位坐标）
    x,y = location

    # 检查该位置是否被置零
    # 防止发送0坐标导致标记进度下降
    if np.isclose(location,0).all():
        send_enable = False
    else:
        send_enable = True
    
    if reverse:
        if send_enable: # 置零便不发送处理
            # 小地图协议，车辆id,float32类型x坐标，float32类型y坐标，串口对象
            if home_test: # 联盟赛5*5场地
                x = 5 - x # X 与spin once中再次中心对称，变回世界坐标系坐标
                y = -(5 + y) # Y
                y = 5+y
            else:
                x = ARER_LENGTH - x # X 与spin once中再次中心对称，变回世界坐标系坐标
                y = -(AREA_WIDTH + y) # Y
                y = AREA_WIDTH+y
            if debug_info:
                print('tiny map x: {} y: {}' .format(x, y))
            Referee_Transmit_Map(0x0305, 14, Id_red, np.float32(x), np.float32(y), ser)
            time.sleep(0.1)
        ControlLoop_red() # 循环红色车编号 1-5 哨兵 7
    else:
        if send_enable:
            if home_test: # 联盟赛5*5场地
                y = 5+y
            else:
                y = AREA_WIDTH+y
            if debug_info:
                print('tiny map x: {} y: {}' .format(x, y))
            Referee_Transmit_Map(0x0305, 14, Id_blue, np.float32(x), np.float32(y), ser) # 红方补给站为0，0
            time.sleep(0.1)
        ControlLoop_blue() # 循环红色车编号 101-105 哨兵 107

    if send_enable:
        UART_passer.loop_send += 1 # 在一轮循环中发送次数

    # 4.24 加入哨兵，循环次数应该是+1
    if ind == 5:
        if  UART_passer.loop_send == 0: # 若一轮循环结束，没有发送一次，则sleep 0.1s来保证不占主线程
            time.sleep(0.1)
        UART_passer.loop_send = 0
    ind = (ind + 1) % 6

# 4.23
def radar_trigger_double_vulnerable():
    Game_radar_cmd.radar_cmd_t = Game_radar_cmd.radar_cmd_t + 1
    temp = Game_radar_cmd.radar_cmd_t
    radar_transmit_double_vulnerable(0x0121, 0x8080, temp, ser)  # 发送给专门的裁判系统

# 4.23 看了半天大概明白了，0x0121作为0x0301的data内容
def radar_transmit_double_vulnerable(dataID, ReceiverId, data, ser):
    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = 10  # 数据帧中 data 的长度,占两个字节
    buffer[2] = 0
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = 0x01
    buffer[6] = 0x03

    # 4.23 以下即为0x0301的data范围，7、8，9、10，11、12
    # 自定义内容ID(0x0200-0x02ff之间)
    buffer[7] = dataID & 0x00ff
    buffer[8] = (dataID & 0xff00) >> 8
    # 发自雷达站 发送者id
    if not reverse:
        buffer[9] = 9
    else:
        buffer[9] = 109
    buffer[10] = 0
    # receiver id
    buffer[11] = ReceiverId & 0x00ff
    buffer[12] = (ReceiverId & 0xff00) >> 8
    # 自定义内容数据段
    # 4.23 传入的应该是int，通信时只给了 1 byte，后面bytearray会统一转，测试了下这里不能data索引也不能转byte
    buffer[13] = data

    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), 10 + 9)  # 整包校验

    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + 10

    for i in range(9 + 10):
        buffer_tmp_array[i] = buffer[i]
    # if debug_info:
        # print('between car alarm:{}' .format(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))


# 5.9 哨兵比较需要标签，那么参考小地图发送的版本
# 其实和小地图的格式很像，但是我感觉集成到Robot_Data_Transmit_Map()内也可以，或者换一个ind
def Tranmit_with_sentry(dataID, ReceiverId, datalength, targetID, x, y, ser):
    buffer = [0]
    buffer = buffer * 200

    buffer[0] = 0xA5  # 数据帧起始字节，固定值为 0xA5
    buffer[1] = (datalength) & 0x00ff  # 数据帧中 data 的长度,占两个字节
    buffer[2] = ((datalength) & 0xff00) >> 8
    buffer[3] = 0  # 包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
    buffer[5] = 0x01
    buffer[6] = 0x03
    # 自定义内容ID(0x0200-0x02ff之间)
    # 0x0210
    buffer[7] = dataID & 0x00ff
    buffer[8] = (dataID & 0xff00) >> 8
    # 发自雷达站
    if not reverse:
        buffer[9] = 9   # 本方作为雷达站
    else:
        buffer[9] = 109
    buffer[10] = 0
    buffer[11] = ReceiverId
    buffer[12] = 0
    # 自定义内容数据段 首先是敌方标签，接着是对应的
    buffer[13] = bytes(x)[0]
    buffer[14] = bytes(x)[1]
    buffer[15] = bytes(x)[2]
    buffer[16] = bytes(x)[3]
    buffer[17] = bytes(y)[0]
    buffer[18] = bytes(y)[1]
    buffer[19] = bytes(y)[2]
    buffer[20] = bytes(y)[3]


    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), datalength + 9)  # 整包校验

    buffer_tmp_array = [0]
    buffer_tmp_array *= 9 + datalength

    for i in range(9 + datalength):
        buffer_tmp_array[i] = buffer[i]
    # if debug_info:
        # print('between car xy:{}' .format(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))
    


# 发送车间通信车辆坐标
def Our_Robot_Coor_Transmit(ser):
    '''
    雷达站发送车间通信包函数
    上限10Hz（需要考虑预警频率）
    '''
    global ind_our_coor
    location = UART_passer.get_our_position()[ind_our_coor] # 得到当前帧位置（来源于主循环中检测出的我方单位坐标）
    x,y = location

    enemy_location = UART_passer.get_position()[ind_our_coor] # 得到当前帧位置（来源于主循环中检测出的敌方单位坐标）
    enemy_x,enemy_y = enemy_location

    # 检查该位置是否被置零
    if np.isclose(location,0).all() and np.isclose(enemy_location,0).all():
        send_enable = False
    else:
        send_enable = True
    
    code = 16 # 注意冲突
    dataID = 0x0200 + (code & 0xFF) # 自定义内容，0x200-0x2ff
    receiverId = 7  # 给哨兵
    if reverse: # 我方车间通信目标转换为（蓝方）
        receiverId += 100

    if send_enable: # 置零便不发送处理
        # print('our_x = {0} our_y = {1}' .format(x, y))
        # print('enemy_x = {0} enemy_y = {1}' .format(enemy_x, enemy_y))
        Referee_Transmit_BetweenCar_Coor(dataID, receiverId, 22, np.float32(x), np.float32(y), np.float32(enemy_x), np.float32(enemy_y), ser)
        time.sleep(0.1)

    if send_enable:
        UART_passer.loop_send_our_coor += 1 # 在一轮循环中发送次数
    if ind_our_coor == 5:
        if  UART_passer.loop_send_our_coor == 0: # 若一轮循环结束，没有发送一次，则sleep 0.1s来保证不占主线程
            time.sleep(0.1)
        UART_passer.loop_send_our_coor = 0
    ind_our_coor = (ind_our_coor + 1) % 6


# 发送车间通信己方车辆坐标(老代码)
# def Our_Robot_Coor_Transmit(ser):
#     '''
#     雷达站发送车间通信包函数
#     上限10Hz（需要考虑预警频率）
#     '''
#     global ind_our_coor
#     location = UART_passer.get_our_position()[ind_our_coor] # 得到当前帧位置（来源于主循环中检测出的我方单位坐标）
#     x,y = location

#     enemy_location = UART_passer.get_position()[ind_our_coor] # 得到当前帧位置（来源于主循环中检测出的敌方单位坐标）
#     enemy_x,enemy_y = enemy_location

#     # 检查该位置是否被置零
#     if np.isclose(location,0).all() and np.isclose(enemy_location,0).all():
#         send_enable = False
#     else:
#         send_enable = True
    
#     code = 16 # 注意冲突
#     dataID = 0x0200 + (code & 0xFF) # 自定义内容，0x200-0x2ff
#     receiverId = 5
#     if reverse: # 我方车间通信目标转换为（蓝方）
#         receiverId += 100

#     if send_enable: # 置零便不发送处理
#         # print('our_x = {0} our_y = {1}' .format(x, y))
#         # print('enemy_x = {0} enemy_y = {1}' .format(enemy_x, enemy_y))
#         Referee_Transmit_BetweenCar_Coor(dataID, receiverId, 22, np.float32(x), np.float32(y), np.float32(enemy_x), np.float32(enemy_y), ser)
#         time.sleep(0.1)

#     if send_enable:
#         UART_passer.loop_send_our_coor += 1 # 在一轮循环中发送次数
#     if ind_our_coor == 4:
#         if  UART_passer.loop_send_our_coor == 0: # 若一轮循环结束，没有发送一次，则sleep 0.1s来保证不占主线程
#             time.sleep(0.1)
#         UART_passer.loop_send_our_coor = 0
#     ind_our_coor = (ind_our_coor + 1) % 5



if __name__ == '__main__':
    Referee_Transmit_Map(0x0305, 14, 103, np.float32(3), np.float32(3), None)