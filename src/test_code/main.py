#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import argparse
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from rknn_pkg.srv import DetectTarget,DetectTargetResponse  

from functools import partial
import threading
import json
import numpy as np
from collections import defaultdict
import time
import threading
import math

from tool import CustomCvBridge
from tool import fit_line_and_calculate_angle


# add path
import sys
import os

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录的路径（parent_dir）
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到系统路径
sys.path.append(parent_dir)

from py_utils.coco_utils import COCO_test_helper

from tool import CustomCvBridge
from tool import post_process,post_process

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5  # 显示阈值

CLASSES = ('banana','cake','cola','apple','lajiao','milk','potato','tomato','greenlight','redlight','watermelon')
TYPE=("fruit","vegetable","dessert","greenlight")

def check_obj(id,aim_type):
    if aim_type == 'fruit':
        return id in [0, 3,10]
    elif aim_type == 'vegetable':
        return id in [4, 6, 7]
    elif aim_type == 'dessert':
        return id in [1, 2, 5]
    elif aim_type == 'greenlight':
        return id in [8]

class YoloDetector:
    def release(self):
        self.model.release()
        self.image_sub.unregister()
        self.scan_sub.unregister()
        self.imu_sub.unregister()
            
    def __init__(self, model_path, target='rk3588', device_id=None, debug=False):
        
        self.debug = debug
        self.aim_type =None
        self.if_parking = None
        self.if_success = False
        self.story_aim = None
        self.time=time.time()
        #sleep->find->pose->close->roat->park
        #分别对应静默状态，寻找目标状态，对齐目标中线状态，靠近摆正状态，停车状态
        self.state ='sleep'
        
        # 加载模型
        self.model, self.platform = self.setup_model(model_path, target, device_id)
        
        # yolo模型前处理后处理的工具
        self.bridge = CustomCvBridge()
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        
        # 图像处理结果，imu数据查询，雷达数据缓存锁
        self.image_obj_pos_lock = threading.Lock()
        self.imu_lock = threading.Lock()
        self.scan_lock = threading.Lock()
        self.processing = False
        
        # 需要订阅的话题
        # 图像数据
        self.image_sub = rospy.Subscriber('/image_rect_color', Image, self.image_callback)
        # 雷达数据
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        # imu数据
        self.imu_sub = rospy.Subscriber('/imu',Imu,self.imu_callback)
        
        # 发布的话题(速度)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 以下是一些全局变量，用来提供给不同的回调函数
        self.cmd_vel_msg = Twist()
        self.cmd_vel_msg.linear.x = 0.0
        self.cmd_vel_msg.linear.y = 0.0
        self.cmd_vel_msg.angular.z = 0.0
        
        # 雷达数据
        self.obstacle_distance = None
        self.obstacle_angle = None
        
        # imu数据
        self.car_pose = None
        
        # 图像处理结果
        self.image_obj_pos = None
        self.image_obj_name = None
        self.input_data = None
        
        # 旋转的累计角度
        self.last_angle = 0.0
        self.total_rotation_angle = 0.0
        
        self.times=0
                
        rospy.loginfo("YOLO 识别节点启动成功。")
    
    def setup_model(self, model_path, target, device_id):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(model_path, target, device_id)
        return model, platform
    
    def image_callback(self, msg):
        """
        图像回调函数
        """
        if self.processing:
            return  # 如果正在处理图像，则忽略当前回调
        self.processing = True  # 设置为正在处理状态
        with self.image_obj_pos_lock:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pad_color = (0, 0, 0)
            img_resized = self.co_helper.letter_box(im=cv_image.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            self.input_data = img_rgb
            
        self.processing = False  # 重置为未处理状态
                
    def imu_callback(self, msg):
        """
        IMU数据回调函数
        """
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
        
        with self.imu_lock:
            x = msg.orientation.x
            y = msg.orientation.y
            z = msg.orientation.z
            w = msg.orientation.w
            # 计算偏航角（弧度）
            yaw_rad = quaternion_to_yaw(x, y, z, w)
            # 转换为角度
            yaw_deg = quaternion_to_degrees(yaw_rad)
            # 在命令行输出
            self.car_pose = yaw_deg
            if self.debug:
                rospy.loginfo(f"二维姿态角度: {yaw_deg:.2f} 度")
        
    
    def scan_msg_postprocess(self,angle_increment, scan_data, intensities):
        """
        处理激光雷达数据,主要是获取到正前方障碍物的距离，以及x轴与正前方障碍物的夹角
        """
        distance =0
        cnt = 0
        n=15
        
        node_list=[]
        
        for i in range(377-12,378+12):
            # 获取原始距离数据
            raw_distance = scan_data[i]
        
            # 过滤异常值：排除inf、nan和超出合理范围的值
            if not (np.isfinite(raw_distance) and -0.001< raw_distance < 10.0):
                print("跳过无效数据:", raw_distance)
                continue  # 跳过无效数据
            
            ag= i * angle_increment- np.pi
            x= scan_data[i]*np.cos(ag)
            y= scan_data[i]*np.sin(ag)
            
            
            
            node_list.append((x,y))
            distance_i = x
            distance += distance_i
            cnt += 1
        
        distance /= cnt 
        angle = fit_line_and_calculate_angle(node_list)
        
        return distance, angle
        
    
    def scan_callback(self, msg):
        """处理激光雷达数据"""
        with self.scan_lock:
            angle_increment = msg.angle_increment
            scan_data = msg.ranges
            intensities = msg.intensities
            distance, angle = self.scan_msg_postprocess(angle_increment, scan_data, intensities)
            self.obstacle_distance = distance
            self.obstacle_angle = angle
            if self.debug:
                rospy.loginfo(f"雷达检测到障碍物距离: {distance:.2f} m, 角度: {angle:.2f} 度")

    def image_postprocess(self):
        outputs = self.model.run([self.input_data])
        boxes, classes, scores = post_process(outputs)

        see_obj = False
        if boxes is not None and len(boxes) > 0:
            # 处理检测结果
            for box,class_id, score in zip(boxes, classes, scores):
                if score < CONFIDENCE_THRESHOLD:
                    continue
                if check_obj(class_id,self.aim_type):
                    obj_pos=(box[0]+box[2])/(2*IMG_SIZE[0])
                    obj_pos=1-obj_pos
                    self.image_obj_pos = obj_pos
                    self.image_obj_name = CLASSES[class_id]
                    self.story_aim = CLASSES[class_id]
                    see_obj = True
                    if self.debug:
                        rospy.loginfo(f"检测到目标: {self.image_obj_name}, 位置: {self.image_obj_pos:.2f}")
                    break
        if not see_obj:
            self.image_obj_pos = None
            self.image_obj_name = None
            if self.debug:
                rospy.loginfo("未检测到目标。")

def runDetector(detector):
    with detector.scan_lock, detector.image_obj_pos_lock, detector.imu_lock:
        if  detector.obstacle_distance is None or detector.car_pose is None :
            if detector.debug:
                rospy.loginfo("小车传感器未就绪，等待数据...")
            return
        
        detector.image_postprocess()
        
        if  detector.state != 'sleep' and detector.state != 'find' and detector.image_obj_pos is None and detector.state != 'park':
            if detector.debug:
                rospy.loginfo("目标丢失，忽略当前帧")
                rospy.loginfo(f"当前状态: {detector.state}")
                rospy.loginfo(f"历史记录: {detector.story_aim}")
            return 'continue'
        
        if detector.debug:
            if detector.image_obj_pos is not None:
                rospy.loginfo(f"检测到目标位置: {detector.image_obj_pos:.2f}, 名称: {detector.image_obj_name}")
            else:
                rospy.loginfo("未检测到目标。")
            
            if detector.obstacle_distance is not None:
                rospy.loginfo(f"障碍物距离: {detector.obstacle_distance:.2f} m, 角度: {detector.obstacle_angle:.2f} 度")
            else:
                rospy.loginfo("未检测到障碍物。")
            
            if detector.car_pose is not None:
                rospy.loginfo(f"车辆姿态角度: {detector.car_pose:.2f} 度")
            else:
                rospy.loginfo("未获取到车辆姿态。")
        
        
        inference_end = time.time()
            
        print(f"延迟{(inference_end - detector.time)*1000:.2f}ms")
        detector.time=inference_end
        
        
        if detector.state == 'sleep':
            if detector.debug:
                rospy.loginfo("当前状态: 静默状态")
            detector.state = 'find'
            detector.last_angle = detector.car_pose
            detector.total_rotation_angle = 0.0
        
        if detector.state == 'find':
            if detector.debug:
                rospy.loginfo("当前状态: 寻找目标状态")
            detector.total_rotation_angle += min(abs(detector.car_pose - detector.last_angle), 360 - abs(detector.car_pose - detector.last_angle))
            detector.last_angle = detector.car_pose
            if detector.total_rotation_angle >= 450:
                rospy.loginfo("已旋转超过450度，当前任务失败。")
                return 'finish'
            
            if detector.image_obj_pos is not None:
                detector.state = 'pose'
                if detector.debug:
                    rospy.loginfo("目标已找到，切换到对齐目标状态。")
            else:
                detector.cmd_vel_msg.linear.x = 0.0
                detector.cmd_vel_msg.linear.y = 0.0
                detector.cmd_vel_msg.angular.z = 0.8  # 继续旋转寻找
                detector.cmd_vel_pub.publish(detector.cmd_vel_msg)
                return 'continue'
        
        # #debug
        # if detector.state == 'pose':
        #     if detector.debug:
        #         rospy.loginfo("当前状态: 对齐目标中线状态")
        #     if detector.image_obj_pos is not None:
        #         # 计算目标位置与图像中心的偏差
        #         deviation = 0.5-detector.image_obj_pos 
        #         detector.cmd_vel_msg.angular.z = 1 * deviation
        #         detector.cmd_vel_msg.linear.x = 0.0 
        #         detector.cmd_vel_msg.linear.y = 0.0
        #         detector.cmd_vel_pub.publish(detector.cmd_vel_msg)
        #         return 'continue'

        # return 'continue'
        
        if detector.state == 'pose':
            if detector.debug:
                rospy.loginfo("当前状态: 对齐目标中线状态")
            if detector.image_obj_pos is not None:
                # 计算目标位置与图像中心的偏差
                deviation = 0.5-detector.image_obj_pos 
                if abs(deviation) < 0.05:
                    # 偏差小于阈值，认为目标已对齐
                    if detector.debug:
                        rospy.loginfo("目标已对齐，切换到靠近摆正状态。")
                    detector.state = 'close'
                else:
                    detector.cmd_vel_msg.angular.z = 1.8 * deviation
                    detector.cmd_vel_msg.linear.x = 0.0 
                    detector.cmd_vel_msg.linear.y = 0.0
                    detector.cmd_vel_pub.publish(detector.cmd_vel_msg)
                    return 'continue'
        
        if detector.state == 'close':
            if detector.debug:
                rospy.loginfo("当前状态: 靠近状态")
            if detector.obstacle_distance < 0.5:
                if detector.debug:
                    rospy.loginfo("已靠近目标，切换到对齐状态。")
                detector.state = 'roat'
            else:
                deviation = 0.5-detector.image_obj_pos
                detector.cmd_vel_msg.linear.x = 0.2
                detector.cmd_vel_msg.linear.y = 0.0
                detector.cmd_vel_msg.angular.z = 1.8 * deviation
                detector.cmd_vel_pub.publish(detector.cmd_vel_msg)
                return 'continue'
        
        if detector.state == 'roat':
            if detector.debug:
                rospy.loginfo("当前状态: 对齐状态")
            if abs(detector.obstacle_angle-np.pi/2) < np.pi/16:
                if detector.debug:
                    rospy.loginfo("已摆正，切换到停车状态。")
                detector.state = 'park'
            else:
                deviation = 0.5-detector.image_obj_pos
                detector.cmd_vel_msg.angular.z = 1.8 * deviation
                detector.cmd_vel_msg.linear.x = 0.0
                detector.cmd_vel_msg.linear.y = 0.1*(np.pi/2- detector.obstacle_angle)
                detector.cmd_vel_pub.publish(detector.cmd_vel_msg)
        
        if detector.state == 'park':
            if detector.if_parking == False:
                detector.if_success = True
                return 'finish'
            if detector.debug:
                rospy.loginfo("当前状态: 停车状态")
            if detector.obstacle_distance < 0.25:
                rospy.loginfo("已停车，任务成功。")
                detector.if_success = True
                return 'finish'
            else:
                detector.cmd_vel_msg.linear.x = 0.2
                detector.cmd_vel_msg.linear.y = 0.0
                detector.cmd_vel_msg.angular.z = 0.0
                detector.cmd_vel_pub.publish(detector.cmd_vel_msg)
                return 'continue'
        
    pass

def service_callback(req, model_path, target, device_id, debug):
    Dector= YoloDetector(model_path, target, device_id, debug)
    Dector.aim_type = req.aim_type
    Dector.if_parking = req.if_parking
    
    rospy.loginfo(f"收到任务请求，目标类型: {Dector.aim_type},是否靠近停车: {'是' if Dector.if_parking else '否'}")
    rate = rospy.Rate(50) 
    while not rospy.is_shutdown():
        
        # 获取各个传感器的数据，为决策做准备
        
        #开始决策
        state=runDetector(Dector)
        if state == 'finish':
            Dector.release()
            return DetectTargetResponse(if_success=Dector.if_success, obj_name=Dector.image_obj_name)
        
        rate.sleep()
    
    rospy.loginfo("服务节点中断。")
    Dector.release()
    # 释放资源
    return DetectTarget.Response(
        if_success=Dector.if_success,
        obj_name=Dector.image_obj_name)

if __name__=='__main__':
    # Service节点的初始化
    rospy.init_node('yolo_detector', anonymous=True)
    model_path = rospy.get_param('~model_path','/home/ucar/ucar_ws/src/rknn_pkg/model/806.rknn') # .rknn模型路径
    target = rospy.get_param('~target', 'rk3588') # 'rk3588'
    device_id = rospy.get_param('~device_id', 0) # 设备ID，默认为0
    debug = rospy.get_param("~debug", True)  # 调试模式，默认关闭
    
    Service_callback_with_params = partial(
        service_callback,
        model_path=model_path,
        target=target,
        device_id=device_id,
        debug=debug
    )
    
    Service = rospy.Service('yolo_tracker', DetectTarget, Service_callback_with_params)
    rospy.loginfo("YOLO Tracker 服务已就绪。")
    
    rospy.spin()