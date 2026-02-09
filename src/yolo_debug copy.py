#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import argparse
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import LaserScan

import threading
import json
import numpy as np
from collections import defaultdict
import time

# --- 新增导入! ---
# 导入我们创建的服务定义和响应
from rknn_pkg.srv import DetectTarget, DetectTargetResponse # <-- 替换 'your_ros_package_name'

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from py_utils.coco_utils import COCO_test_helper
from tool import CustomCvBridge, post_process

# 常量
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5
CLASSES = ('banana', 'cake', 'cola', 'apple', 'lajiao', 'milk', 'potato', 'tomato', 'greenlight', 'redlight', 'watermelon')
CLASS_COLORS = { 'banana': (255, 255, 0), 'cake': (255, 128, 0), 'cola': (0, 255, 0), 'apple': (0, 128, 255), 'lajiao': (0, 0, 255), 'milk': (255, 255, 255), 'potato': (255, 0, 0), 'tomato': (255, 0, 255), 'greenlight': (128, 255, 0), 'redlight': (0, 255, 255), 'watermelon': (128, 0, 255) }

class YoloObjectFinderService:
    def __init__(self, model_path, target='rk3588', device_id=None):
        # 状态和控制变量
        self.state = 'idle'  # 状态: idle, finding, tracking, parking, fail, success
        self.aim_type = None # 将由服务请求设置
        self.service_active = threading.Event() # 用于控制服务活动的事件
        
        # 存储服务结果的变量
        self.detected_object_name = ""
        self.final_distance = 0.0

        self.begin_angle = 0.0
        self.counter = 0
        self.rotation_direction = 1
        
        # 模型初始化
        self.model, self.platform = self.setup_model(model_path, target, device_id)
        
        # ROS组件
        self.bridge = CustomCvBridge()
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        
        self.image_lock = threading.Lock()
        self.lock = threading.Lock()
        self.processing = False
        self.theta = 0.0
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_msg = Twist()
        
        self.scan_lock = threading.Lock()
        self.scan_data = None
        self.intensities = None
        self.angle_increment = None
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        self.image_sub = rospy.Subscriber('/image_rect_color', Image, self.image_callback)
        self.annotated_pub = rospy.Publisher('/yolo_detection/annotated_image', Image, queue_size=1)
        self.mag_pose_sub = rospy.Subscriber('/mag_pose_2d', Pose2D, self.mag_pose_callback, queue_size=10)

        # PID参数
        self.Kp_angular, self.Ki_angular, self.Kd_angular = 1.0, 0.01, 0.0
        self.integral_angular, self.last_error_angular = 0.0, 0.0
        self.max_angular_vel = 1.0
        
        self.Kp_linear, self.Ki_linear, self.Kd_linear = 0.6, 0.005, 0.0
        self.integral_linear, self.last_error_linear = 0.0, 0.0
        self.target_distance = 0.2
        self.max_linear_vel = 0.3

        self.Kp_lateral, self.Ki_lateral, self.Kd_lateral = 0.5, 0.0, 0.0
        self.integral_lateral, self.last_error_lateral = 0.0, 0.0
        self.max_lateral_vel = 0.1
        
        self.last_time = rospy.Time.now()
        
        # --- 新增ROS服务 ---
        self.service = rospy.Service('find_object_service', DetectTarget, self.handle_find_object_request)

        rospy.loginfo("YOLO对象查找服务已初始化。")
        rospy.loginfo(f"模型平台: {self.platform}")

    def handle_find_object_request(self, req):
        """
        当接收到新的服务请求时，此函数将被调用。
        它启动状态机并等待其完成。
        """
        rospy.loginfo(f"收到查找对象类型为: {req.target_type} 的服务请求。")
        
        # 重置状态和结果变量
        self.aim_type = req.target_type
        self.detected_object_name = ""
        self.final_distance = 0.0
        self.state = 'finding' # 启动状态机
        self.service_active.set() # 激活图像回调中的处理

        with self.lock:
            self.begin_angle = self.theta
        self.counter = 0
        self.rotation_direction = 1
        
        # 等待状态机达到最终状态
        while self.state not in ['success', 'fail'] and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # 停止机器人
        self.cmd_vel_msg.linear.x = 0
        self.cmd_vel_msg.linear.y = 0
        self.cmd_vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(self.cmd_vel_msg)
        
        # 构建并返回响应
        response = DetectTargetResponse()
        if self.state == 'success':
            response.success = True
            response.detected_object_name = self.detected_object_name
            response.final_distance = self.final_distance
            rospy.loginfo("服务请求完成: 成功!")
        else:
            response.success = False
            rospy.loginfo("服务请求完成: 失败!")

        # 为下一个请求将状态重置为idle
        self.state = 'idle'
        self.service_active.clear()

        return response

    def check_obj(self, id):
        if self.aim_type == 'fruit':
            return id in [0, 3, 10]
        elif self.aim_type == 'vegetable':
            return id in [4, 6, 7]
        elif self.aim_type == 'dessert':
            return id in [1, 2, 5]
        return False

    def setup_model(self, model_path, target, device_id):
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(model_path, target, device_id)
        return model, 'rknn'
    
    def mag_pose_callback(self, msg):
        with self.lock:
            self.theta = msg.theta
    
    def scan_callback(self, msg):
        with self.scan_lock:
            self.angle_increment = msg.angle_increment
            self.scan_data = msg.ranges
            self.intensities = msg.intensities
    
    def image_callback(self, msg):
        # 只有当服务激活时才处理
        if not self.service_active.is_set() or self.processing:
            return
            
        self.processing = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_and_control(cv_image)
        except Exception as e:
            rospy.logerr(f"图像处理错误: {e}")
        finally:
            self.processing = False
    
    def process_and_control(self, img_src):
        # 此函数的大部分逻辑保持不变，只是在处理 'success' 和 'fail' 状态时略有变化。
        try:
            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time

            # 图像预处理
            img_resized = self.co_helper.letter_box(im=img_src.copy(), new_shape=IMG_SIZE, pad_color=(0,0,0))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # 模型推理
            outputs = self.model.run([img_rgb])
            boxes, classes, scores = post_process(outputs)
            
            rospy.loginfo(f"当前状态: {self.state}")

            # 状态机
            if self.state == 'finding':
                rotation_angle_threshold = np.pi / 8
                max_small_rotations = 16 

                with self.lock:
                    current_angle_diff = (self.theta - self.begin_angle + 2*np.pi) % (2*np.pi)
                
                if current_angle_diff > rotation_angle_threshold or current_angle_diff < -rotation_angle_threshold:
                    self.counter += 1
                    self.begin_angle = self.theta
                
                if self.counter >= max_small_rotations:
                    self.state = 'fail'
                    return # 退出以允许服务处理程序响应

                self.cmd_vel_msg.angular.z = 0.5 * self.rotation_direction
                self.cmd_vel_pub.publish(self.cmd_vel_msg)

            elif self.state == 'parking':
                # ... (距离PID逻辑相同)
                current_distance = 0.0
                with self.scan_lock:
                    if self.scan_data:
                        center_idx = len(self.scan_data) // 2
                        scan_range = self.scan_data[center_idx-15:center_idx+15]
                        valid_distances = [d for d in scan_range if 0.01 < d < 5.0]
                        if valid_distances: current_distance = np.mean(valid_distances)
                        else: current_distance = 10.0
                
                self.distance = current_distance
                distance_error = self.distance - self.target_distance
                
                if abs(distance_error) < 0.1: # 停车成功条件
                    self.cmd_vel_msg.linear.x = 0
                    self.cmd_vel_pub.publish(self.cmd_vel_msg)
                    # --- 重要更改 ---
                    # 不关闭，而是将状态设置为成功并存储结果
                    self.final_distance = self.distance
                    self.state = 'success' 
                    return # 退出以允许服务处理程序响应
                else:
                    # ... 线性速度PID计算 ...
                    self.integral_linear += distance_error * dt
                    self.integral_linear = np.clip(self.integral_linear, -0.5, 0.5)
                    derivative_linear = (distance_error - self.last_error_linear) / dt if dt > 0 else 0
                    self.last_error_linear = distance_error
                    linear_vel_output = self.Kp_linear * distance_error + self.Ki_linear * self.integral_linear + self.Kd_linear * derivative_linear
                    self.cmd_vel_msg.linear.x = np.clip(linear_vel_output, -self.max_linear_vel, self.max_linear_vel)
                    self.cmd_vel_pub.publish(self.cmd_vel_msg)

            # 对象检测逻辑
            See_obj = False
            if boxes is not None and len(boxes) > 0:
                for box, class_id, score in zip(boxes, classes, scores):
                    if score >= CONFIDENCE_THRESHOLD and self.check_obj(class_id):
                        See_obj = True
                        obj_pos = ((box[0] + box[2]) / 2.0 / IMG_SIZE[0]) - 0.5
                        # --- 重要更改 ---
                        # 存储检测到的对象名称
                        self.detected_object_name = CLASSES[class_id]
                        break

            if self.state == 'finding' and See_obj:
                self.state = 'tracking'
                rospy.loginfo("检测到对象。切换到跟踪状态。")
            
            if self.state == 'tracking':
                if See_obj:
                    # PID跟踪逻辑 (与之前相同)
                    angular_error = obj_pos
                    # ... (角度PID计算) ...
                    angular_vel_output = angular_error # 假设PID已计算
                    
                    # 距离逻辑以决定是否停车
                    # ... (current_distance_front 的计算) ...
                    current_distance_front = 10.0 # 占位符，需要根据你的扫描数据计算
                    with self.scan_lock:
                        if self.scan_data and self.angle_increment:
                            center_idx = len(self.scan_data) // 2
                            scan_range_start = max(0, center_idx - 15)
                            scan_range_end = min(len(self.scan_data), center_idx + 15)
                            valid_distances_front = [d for d in self.scan_data[scan_range_start:scan_range_end] if 0.01 < d < 5.0]
                            if valid_distances_front: current_distance_front = np.mean(valid_distances_front)
                    self.distance = current_distance_front # 更新实例变量

                    if self.distance <= self.target_distance + 0.1:
                         self.state = 'parking'
                         rospy.loginfo("接近目标距离。切换到停车状态。")
                         self.cmd_vel_msg.linear.x = 0
                    else:
                         self.cmd_vel_msg.linear.x = self.max_linear_vel

                    self.cmd_vel_msg.angular.z = angular_vel_output
                    self.cmd_vel_pub.publish(self.cmd_vel_msg)
                else:
                    # 跟踪过程中丢失对象
                    self.state = 'finding' # 回到查找
                    rospy.loginfo("对象丢失。返回查找状态。")
        
        except Exception as e:
            rospy.logerr(f"process_and_control 错误: {e}")
            self.state = 'fail' # 如果发生错误，服务失败

    def run(self):
        """保持节点运行。"""
        rospy.spin()
    
    def shutdown(self):
        self.model.release()
        rospy.loginfo("YOLO对象查找服务关闭")


if __name__ == '__main__':
    try:
        rospy.init_node('yolo_object_finder_service', anonymous=False)

        model_path = rospy.get_param('~model_path')
        target = rospy.get_param('~target', 'rk3588')
        device_id = rospy.get_param('~device_id', 0)

        # 创建 YoloObjectFinderService 实例
        finder_service = YoloObjectFinderService(model_path, target, device_id)
        
        rospy.on_shutdown(finder_service.shutdown)
        
        # 服务服务器在 __init__ 中启动。
        # rospy.spin() 使节点保持活动状态，监听请求。
        rospy.loginfo("主节点正在等待服务请求...")
        finder_service.run()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"主节点发生致命错误: {e}")