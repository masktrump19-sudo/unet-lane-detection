#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import argparse
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan

import threading
import json
import numpy as np
from collections import defaultdict
import time
import threading


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

TYPE=("fruit","vegetable","dessert")


aim_type = 'vegetable'  # 目标类型

def check_obj(id):
    global aim_type
    if aim_type == 'fruit':
        return id in [0, 3,10]
    elif aim_type == 'vegetable':
        return id in [4, 6, 7]
    elif aim_type == 'dessert':
        return id in [1, 2, 5]


class YoloContinuousDetector:
    def __init__(self, model_path, target='rk3588', device_id=None):
        self.state='sleep'
        self.begin_angle=0.0
        self.counter = 0
        # Initialize model
        self.model, self.platform = self.setup_model(model_path, target, device_id)
        
        # Initialize ROS components
        self.bridge = CustomCvBridge()
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        
        # 图像缓存和同步
        self.image_lock = threading.Lock()
        self.lock = threading.Lock()
        self.processing = False
        self.theta=0.0
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_msg = Twist()
        self.cmd_vel_msg.linear.x = 0.0
        self.cmd_vel_msg.linear.y = 0.0
        self.cmd_vel_msg.angular.z = 0.0
        
        self.scan_lock = threading.Lock()
        self.scan_data = None
        self.intensities= None
        self.angle_increment = None
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # ROS subscribers and publishers
        self.image_sub = rospy.Subscriber('/image_rect_color', Image, self.image_callback)
        self.annotated_pub = rospy.Publisher('/yolo_detection/annotated_image', Image, queue_size=1)
        self.mag_pose_sub = rospy.Subscriber(
            '/mag_pose_2d',
            Pose2D,
            self.mag_pose_callback,
            queue_size=10
        )
        
        rospy.loginfo("YOLO Continuous Detector initialized")
        rospy.loginfo(f"Model platform: {self.platform}")
        rospy.loginfo("Publishing annotated images to '/yolo_detection/annotated_image'")
    
    def setup_model(self, model_path, target, device_id):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(model_path, target, device_id)
        return model, platform
    
    def mag_pose_callback(self, msg):
        """处理磁力计姿态数据"""
        with self.lock:
            self.theta = msg.theta
            #rospy.loginfo(f"Received mag pose: theta={self.theta}")
    
    def scan_callback(self, msg):
        """处理激光雷达数据"""
        with self.scan_lock:
            self.angle_increment = msg.angle_increment
            self.scan_data = msg.ranges
            self.intensities = msg.intensities
            #rospy.loginfo("Received scan data")
    
    def image_callback(self, msg):
        # 避免重复处理
        if self.processing:
            return
            
        self.processing = True
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 处理图像并发布结果
            self.process_and_annotate_image(cv_image)
            
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
        finally:
            self.processing = False
    
    def process_and_annotate_image(self, img_src):
        """处理图像并添加标注"""
        try:
            # 保存原始图像尺寸
            orig_height, orig_width = img_src.shape[:2]
            
            # Preprocess image for inference
            pad_color = (0, 0, 0)
            img_resized = self.co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_data = img_rgb
            
            # 开始计时
            inference_start = time.time()
            
            # Run inference
            outputs = self.model.run([input_data])
            boxes, classes, scores = post_process(outputs)
            #print(boxes, classes, scores)
            print(self.state)
            if self.state=='sleep':
                self.state='finding'
                with self.lock:
                    self.begin_angle = self.theta
            elif self.state=='finding':
                if (2*np.pi+self.theta - self.begin_angle)%(2*np.pi)> np.pi/4:
                    self.counter += 1
                    self.begin_angle = self.theta
                    if self.counter >= 12:  # 每旋转45度检测一次
                        self.state='fail'

            print(self.counter)
            if self.state=='fail':
                rospy.signal_shutdown("目标物体寻找失败，退出")
            if self.state=='parking':
                with self.scan_lock:
                    if self.scan_data is not None and self.angle_increment is not None:
                        cnt_1=0
                        cnt_2=0
                        tot_1=0.0
                        tot_2=0.0
                        for i in range(377-15,377+15):
                            if self.intensities[i] > 1000.0:
                                if i<377:
                                    cnt_1 += 1
                                    tot_1+= self.scan_data[i]
                                else:
                                    cnt_2 += 1
                                    tot_2+= self.scan_data[i]
                        if cnt_1+cnt_2 == 0:
                            self.distance = 0
                        else:
                            self.distance = (tot_1+tot_2) / (cnt_1+cnt_2)
                if self.distance > 0.2:
                    self.cmd_vel_msg.linear.x = 0.1
                    self.cmd_vel_msg.linear.y = 0.0
                    self.cmd_vel_msg.angular.z = 0.0
                    self.cmd_vel_pub.publish(self.cmd_vel_msg)
                    self.state='success'
                else:
                    rospy.signal_shutdown("目标物体停车成功，退出")
                
            # 结束计时并输出
            inference_end = time.time()
            inference_time = (inference_end - inference_start) * 1000  # 转换为毫秒
            print(f"Inference time: {inference_time:.2f} ms")
            See_obj=False
            obj_pos=None
            if boxes is not None and len(boxes) > 0:
                for box,class_id, score in zip(boxes, classes, scores):
                    if score < CONFIDENCE_THRESHOLD:
                        continue
                    if check_obj(class_id):
                        See_obj=True
                        obj_pos=(box[0]+box[2])/(2*IMG_SIZE[0])
                        obj_pos=1-obj_pos
                        print(f"目标物体位置: {obj_pos}")
                        break
            if See_obj:
                self.state='tracking'
                dis=0.5- obj_pos
                self.distance=0
                with self.scan_lock:
                    if self.scan_data is not None and self.angle_increment is not None:
                        cnt_1=0
                        cnt_2=0
                        tot_1=0.0
                        tot_2=0.0
                        for i in range(377-15,377+15):
                            if self.intensities[i] > 1000.0:
                                if i<377:
                                    cnt_1 += 1
                                    tot_1+= self.scan_data[i]
                                else:
                                    cnt_2 += 1
                                    tot_2+= self.scan_data[i]
                        print(f"左侧点数: {cnt_1}, 右侧点数: {cnt_2}")
                        if cnt_1+cnt_2 == 0:
                            self.distance = 0
                        else:
                            self.distance = (tot_1+tot_2) / (cnt_1+cnt_2) 
                        if self.distance >0.5:
                            self.cmd_vel_msg.linear.x = 0.2
                            self.cmd_vel_msg.linear.y = 0.0
                        else:
                            print("检测差别：",abs(tot_2/cnt_2- tot_1/cnt_1))
                            if abs(tot_2/cnt_2- tot_1/cnt_1)<0.006:
                                self.state='parking'
                            else:
                                self.cmd_vel_msg.linear.x = 0.0
                                self.cmd_vel_msg.linear.y = 3*(tot_2/cnt_2- tot_1/cnt_1)
                        self.cmd_vel_msg.angular.z = 2*dis
                        self.cmd_vel_pub.publish(self.cmd_vel_msg)
            else:
                rospy.loginfo("目标物体未检测到,旋转寻找")
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.linear.y = 0.0
                self.cmd_vel_msg.angular.z =1.0
                self.cmd_vel_pub.publish(self.cmd_vel_msg)
            
            
        except Exception as e:
            rospy.logerr(f"Error processing and annotating image: {e}")
            return None
    
    def run(self):
        rospy.spin()
    
    def shutdown(self):
        self.model.release()
        rospy.loginfo("YOLO Continuous Detector shutdown")


if __name__ == '__main__':
    rospy.init_node('yolo_debug', anonymous=False)

    # 获取参数（ROS参数服务器方式）
    model_path = rospy.get_param('~model_path','/home/ucar/ucar_ws/src/rknn_pkg/model/806.rknn')
    target = rospy.get_param('~target', 'rk3588')
    device_id = rospy.get_param('~device_id', 0)

    try:
        detector = YoloContinuousDetector(model_path,target, device_id)
        rospy.on_shutdown(detector.shutdown)
        detector.run()
    except rospy.ROSInterruptException:
        pass