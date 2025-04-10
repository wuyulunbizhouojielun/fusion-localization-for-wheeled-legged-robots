#!/usr/bin/python3
# coding = utf-8
'''
官方demo文档
'''

# 4.23 这里的每个变量都没有指定具体的类型，估计是运行的时候指定

import numpy as np

# 4.23 ok
class game_state:
    def __init__(self):
        game_state.game_type = 4
        game_state.game_progress = 4
        game_state.stage_remain_time = [0, 0]    # 官方协议里是16位无符号

# 4.23 ok
class game_result:
    def __init__(self):
        game_result.winner = 0

# 4.23 ok
class game_robot_HP:
    def __init__(self):
        game_robot_HP.red_1_robot_HP = [0, 0]
        game_robot_HP.red_2_robot_HP = [0, 0]
        game_robot_HP.red_3_robot_HP = [0, 0]
        game_robot_HP.red_4_robot_HP = [0, 0]
        game_robot_HP.red_5_robot_HP = [0, 0]
        game_robot_HP.red_7_robot_HP = [0, 0]
        game_robot_HP.red_outpost_HP = [0, 0]
        game_robot_HP.red_base_HP = [0, 0]
        game_robot_HP.blue_1_robot_HP = [0, 0]
        game_robot_HP.blue_2_robot_HP = [0, 0]
        game_robot_HP.blue_3_robot_HP = [0, 0]
        game_robot_HP.blue_4_robot_HP = [0, 0]
        game_robot_HP.blue_5_robot_HP = [0, 0]
        game_robot_HP.blue_7_robot_HP = [0, 0]
        game_robot_HP.blue_outpost_HP = [0, 0]
        game_robot_HP.blue_base_HP = [0, 0]

# 飞镖状态
# 4.23 上赛季已弃用
class dart_status:
    def __init__(self):
        dart_status.dart_belong = 0
        dart_status.stage_remaining_time = [0, 0]

# 4.23 ok
class event_data:
    def __init__(self):
        event_data.event_type = [0, 0, 0, 0]

# 补弹相关
# 4.23 站口id变成保留字了
class supply_projectile_action:
    def __init__(self):
        # supply_projectile_action.supply_projectile_id = 0
        supply_projectile_action.reserved = 0
        supply_projectile_action.supply_robot_id = 0
        supply_projectile_action.supply_projectile_step = 0
        supply_projectile_action.supply_projectile_num = 0


# 4.23 新增判罚等级违规次数
class refree_warning:
    def __init__(self):
        refree_warning.level = 0
        refree_warning.foul_robot_id = 0
        refree_warning.count = 0

# 飞镖剩余时间
# 4.23 对应0x0105，新增dart_info，方便起见修改结构
# class dart_remaining_time:
#     def __init__(self):
#         dart_remaining_time.time = 0
class dart_info_t:
    def  __init__(self):
        dart_info_t.dart_remaining_time = 0
        dart_info_t.dart_info = [0,0]
 
# 4.23 好像没用上
class custom_data0:
    def _init_(self):
        custom_data0.data1 = [0, 0, 0, 0]
        custom_data0.data2 = [0, 0, 0, 0]
        custom_data0.data3 = [0, 0, 0, 0]
        custom_data0.masks = 0

# 4.23 好像没用上
class graphic_data_struct:
    def __init__(self):
        graphic_data_struct.graphic_name = [0, 0, 0]
        graphic_data_struct.operate_tpye = [0, 0, 0, 0]
        graphic_data_struct.graphic_tpye = [0, 0, 0, 0]
        graphic_data_struct.layer = [0, 0, 0, 0]
        graphic_data_struct.color = [0, 0, 0, 0]
        graphic_data_struct.start_angle = [0, 0, 0, 0]
        graphic_data_struct.end_angle = [0, 0, 0, 0]
        graphic_data_struct.width = [0, 0, 0, 0]
        graphic_data_struct.start_x = [0, 0, 0, 0]
        graphic_data_struct.start_y = [0, 0, 0, 0]
        graphic_data_struct.radius = [0, 0, 0, 0]
        graphic_data_struct.end_x = [0, 0, 0, 0]
        graphic_data_struct.end_y = [0, 0, 0, 0]

    def Add(self):
        graphic_data_struct.data = []
        graphic_data_struct.datalength = 15

# 4.23 似乎没有import过去
class robot_location():
    # the global variables for this class
    loc = np.zeros((6,2),dtype=np.float32)
    def __init__(self):
        pass
    @staticmethod
    def push(location):
        robot_location.loc = np.float32(location)

# 4.23 本赛季新增雷达自主决策指令信息，一个是和裁判系统同步，另一个感觉得包装成数据帧发送
class radar_info_t:
    def __init__(self):
        radar_info_t.radar_info = 0

class radar_cmd_t:
    def __init__(self):
        radar_info_t.radar_cmd = 0

# 4.26 各个机器人标记进度数据
class mark_data_t:
    def __init__(self):
        mark_data_t.mark_hero_progress = 0
        mark_data_t.mark_engineer_progress = 0
        mark_data_t.mark_standard_3_progress = 0
        mark_data_t.mark_standard_4_progress = 0
        mark_data_t.mark_standard_5_progress = 0
        mark_data_t.mark_sentry_progress = 0



