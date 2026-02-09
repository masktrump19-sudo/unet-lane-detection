#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
from sensor_msgs.msg import Imu

def quaternion_to_yaw(x, y, z, w):
    """
    将四元数转换为偏航角（yaw），单位为弧度
    偏航角表示绕z轴的旋转，对应二维平面内的旋转角度
    """
    # 计算偏航角 (yaw)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return yaw

def quaternion_to_degrees(yaw_rad):
    """将弧度转换为角度，范围调整为0-360度"""
    yaw_deg = math.degrees(yaw_rad)
    # 将角度范围从[-180, 180]转换为[0, 360]
    if yaw_deg < 0:
        yaw_deg += 360
    return yaw_deg

def imu_callback(msg):
    """IMU消息回调函数"""
    # 从IMU消息中获取四元数
    x = msg.orientation.x
    y = msg.orientation.y
    z = msg.orientation.z
    w = msg.orientation.w
    
    # 计算偏航角（弧度）
    yaw_rad = quaternion_to_yaw(x, y, z, w)
    # 转换为角度
    yaw_deg = quaternion_to_degrees(yaw_rad)
    
    # 在命令行输出
    rospy.loginfo(f"二维姿态角度: {yaw_deg:.2f} 度")

def imu_listener():
    """初始化节点并订阅IMU话题"""
    # 初始化节点
    rospy.init_node('imu_yaw_listener', anonymous=True)
    
    # 订阅IMU话题，默认话题名为'imu'，可根据实际情况修改
    # 回调函数为imu_callback
    rospy.Subscriber('imu', Imu, imu_callback)
    
    # 保持节点运行，等待回调
    rospy.spin()

if __name__ == '__main__':
    try:
        imu_listener()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
