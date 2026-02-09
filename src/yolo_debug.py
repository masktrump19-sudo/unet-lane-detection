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

# 目标检测的置信度阈值
OBJ_THRESH = 0.25
# 非极大值抑制的阈值
NMS_THRESH = 0.45
# 图像尺寸，用于模型输入
IMG_SIZE = (640, 640)
# 显示检测结果的置信度阈值
CONFIDENCE_THRESHOLD = 0.5  # 显示阈值

# 定义YOLO模型检测的类别
CLASSES = ('banana','cake','cola','apple','lajiao','milk','potato','tomato','greenlight','redlight','watermelon')

# 定义目标类型
TYPE=("fruit","vegetable","dessert")


aim_type = 'dessert'  # 目标类型，这里设置为'dessert'

def check_obj(id):
    """
    根据目标类型检查对象ID是否符合要求
    Args:
        id (int): 检测到的对象的类别ID

    Returns:
        bool: 如果对象ID符合目标类型，则返回True，否则返回False
    """
    global aim_type
    if aim_type == 'fruit':
        return id in [0, 3,10] # banana, apple, watermelon
    elif aim_type == 'vegetable':
        return id in [4, 6, 7] # lajiao, potato, tomato
    elif aim_type == 'dessert':
        return id in [1, 2, 5] # cake, cola, milk

# COCO数据集的ID列表，但在这个代码中似乎没有直接使用，可能用于其他调试或扩展
coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


# 类别颜色映射，用于在图像上绘制不同类别的检测框
CLASS_COLORS = {
    'banana': (255, 255, 0),    # 黄色
    'cake': (255, 128, 0),      # 橙色
    'cola': (0, 255, 0),        # 绿色
    'apple': (0, 128, 255),     # 浅蓝色
    'lajiao': (0, 0, 255),      # 蓝色
    'milk': (255, 255, 255),    # 白色
    'potato': (255, 0, 0),      # 红色
    'tomato': (255, 0, 255),    # 品红色
    'greenlight': (128, 255, 0),# 绿黄色
    'redlight': (0, 255, 255),  # 青色
    'watermelon': (128, 0, 255) # 紫色
}


class YoloContinuousDetector:
    """
    YOLO连续检测器类，用于处理ROS图像流，进行目标检测，并根据检测结果控制机器人运动。
    """
    def __init__(self, model_path, target='rk3588', device_id=None):
        """
        初始化YoloContinuousDetector。
        Args:
            model_path (str): YOLO模型的路径。
            target (str): 部署模型的硬件平台（如'rk3588'）。
            device_id (int): 设备ID，如果存在多个设备。
        """
        self.state='sleep' # 机器人当前状态：sleep, finding, tracking, parking, fail, success
        self.begin_angle=0.0 # 用于记录开始旋转时的角度
        self.counter = 0 # 旋转计数器
        self.rotation_direction = 1 # 旋转方向：1为顺时针，-1为逆时针
        
        # 初始化模型
        self.model, self.platform = self.setup_model(model_path, target, device_id)
        
        # 初始化ROS组件
        self.bridge = CustomCvBridge() # 用于OpenCV图像和ROS图像消息之间的转换
        self.co_helper = COCO_test_helper(enable_letter_box=True) # 用于图像预处理（如letterbox缩放）
        
        # 图像缓存和同步锁，用于线程安全
        self.image_lock = threading.Lock()
        self.lock = threading.Lock()
        self.processing = False # 标志位，表示是否正在处理图像
        self.theta=0.0 # 机器人当前姿态（角度）
        
        # ROS发布器，用于控制机器人速度
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.cmd_vel_msg = Twist() # 速度控制消息
        self.cmd_vel_msg.linear.x = 0.0
        self.cmd_vel_msg.linear.y = 0.0
        self.cmd_vel_msg.angular.z = 0.0
        
        # 激光雷达数据锁和数据
        self.scan_lock = threading.Lock()
        self.scan_data = None # 激光雷达距离数据
        self.intensities= None # 激光雷达强度数据
        self.angle_increment = None # 激光雷达角度增量
        rospy.Subscriber('/scan', LaserScan, self.scan_callback) # 订阅激光雷达话题
        
        # ROS订阅器和发布器
        self.image_sub = rospy.Subscriber('/image_rect_color', Image, self.image_callback) # 订阅相机图像话题
        self.annotated_pub = rospy.Publisher('/yolo_detection/annotated_image', Image, queue_size=1) # 发布标注后的图像
        self.mag_pose_sub = rospy.Subscriber(
            '/mag_pose_2d',
            Pose2D,
            self.mag_pose_callback,
            queue_size=10
        ) # 订阅磁力计姿态话题

        # PID控制器参数
        # 角速度PID参数
        self.Kp_angular = 0.5  # 比例增益 (可以尝试 1.8-2.5)
        self.Ki_angular = 0.01 # 积分增益 (可以尝试 0.005-0.02)
        self.Kd_angular = 0.0  # 微分增益 (可以尝试 0.05-0.1, 但要小心振荡)
        self.integral_angular = 0.0
        self.last_error_angular = 0.0
        self.max_angular_vel = 1.0 # 最大角速度
        
        # 线速度PID参数 (用于距离控制)
        self.Kp_linear = 0.6   # 比例增益 (可以尝试 0.5-0.8)
        self.Ki_linear = 0.005 # 积分增益 (可以尝试 0.002-0.01)
        self.Kd_linear = 0.0  # 微分增益
        self.integral_linear = 0.0
        self.last_error_linear = 0.0
        self.target_distance = 0.2 # 目标停车距离 (0.2米)
        self.max_linear_vel = 0.3 # 最大线速度

        # 横向速度PID参数 (用于侧向调整)
        self.Kp_lateral = 1.0 # 比例增益 (可以尝试 0.6-1.0)
        self.Ki_lateral = 0.0 # 积分增益 (可以尝试 0.001-0.005)
        self.Kd_lateral = 0.0 # 微分增益
        self.integral_lateral = 0.0
        self.last_error_lateral = 0.0
        self.max_lateral_vel = 0.1# 最大横向速度
        
        self.last_time = rospy.Time.now() # 用于计算PID的dt

        rospy.loginfo("YOLO Continuous Detector initialized")


        rospy.loginfo(f"Model platform: {self.platform}")
        rospy.loginfo("Publishing annotated images to '/yolo_detection/annotated_image'")
    
    def setup_model(self, model_path, target, device_id):
        """
        设置模型，根据目标平台加载RKNN模型。
        Args:
            model_path (str): 模型文件路径。
            target (str): 目标硬件平台。
            device_id (int): 设备ID。

        Returns:
            tuple: 包含模型实例和平台名称的元组。
        """
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(model_path, target, device_id)
        return model, platform
    
    def mag_pose_callback(self, msg):
        """
        处理磁力计姿态数据。
        Args:
            msg (Pose2D): 磁力计姿态消息。
        """
        with self.lock:
            self.theta = msg.theta # 更新机器人当前角度
            # rospy.loginfo(f"Received mag pose: theta={self.theta}") # 避免频繁打印
    
    def scan_callback(self, msg):
        """
        处理激光雷达数据。
        Args:
            msg (LaserScan): 激光雷达扫描消息。
        """
        with self.scan_lock:
            self.angle_increment = msg.angle_increment # 角度增量
            self.scan_data = msg.ranges # 距离数据
            self.intensities = msg.intensities # 强度数据
            # rospy.loginfo("Received scan data") # 避免频繁打印
    
    def image_callback(self, msg):
        """
        图像话题的回调函数，处理接收到的图像。
        Args:
            msg (Image): ROS图像消息。
        """
        # 避免重复处理，如果当前正在处理图像，则直接返回
        if self.processing:
            return
            
        self.processing = True # 设置处理标志为True
        
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 处理图像并发布结果
            self.process_and_annotate_image(cv_image)
            
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
        finally:
            self.processing = False # 处理完成后，将标志设为False
    
    def process_and_annotate_image(self, img_src):
        """
        处理图像并添加标注。
        Args:
            img_src (numpy.ndarray): 原始OpenCV图像。
        """
        try:
            # 获取当前时间，用于PID控制器时间步计算
            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time

            # 保存原始图像尺寸
            orig_height, orig_width = img_src.shape[:2]
            
            # 图像预处理，进行letterbox缩放
            pad_color = (0, 0, 0) # 填充颜色
            img_resized = self.co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # 转换为RGB格式

            input_data = img_rgb # 模型输入数据
            
            # 开始计时，测量推理时间
            inference_start = time.time()
            
            # 运行模型推理
            outputs = self.model.run([input_data])
            # 对模型输出进行后处理，得到边界框、类别和分数
            boxes, classes, scores = post_process(outputs)
            
            # 获取用于标注的原始图像副本
            img_display = img_src.copy() 

            #print(boxes, classes, scores)
            print(f"当前状态: {self.state}") # 打印当前机器人状态

            # 根据状态机进行逻辑判断和机器人控制
            if self.state=='sleep':
                self.state='finding' # 从sleep状态切换到finding状态
                with self.lock:
                    self.begin_angle = self.theta # 记录开始寻找时的角度
                self.counter = 0 # 重置寻找计数器
                self.rotation_direction = 1 # 初始旋转方向为顺时针
                rospy.loginfo("进入寻找状态...")

            elif self.state=='finding':
                # 更智能的寻找策略：旋转一段角度，如果未发现目标，则切换方向或进行更大范围的旋转
                rotation_angle_threshold = np.pi / 8 # 每次旋转约22.5度
                max_small_rotations = 16 # 允许的小角度旋转次数 (16 * 22.5 = 360度)

                with self.lock:
                    current_angle_diff = (self.theta - self.begin_angle + 2*np.pi) % (2*np.pi)
                
                # 检查是否已旋转过一个阈值角度
                if abs(current_angle_diff) > rotation_angle_threshold:
                    self.counter += 1
                    self.begin_angle = self.theta # 更新起始角度
                    rospy.loginfo(f"寻找状态，已旋转 {self.counter} 次")

                if self.counter >= max_small_rotations:
                    rospy.loginfo("完成一圈小角度旋转，仍未发现目标。")
                    self.state='fail' # 寻找失败
                    rospy.signal_shutdown("目标未找到，退出。") # 如果找不到目标，可以考虑退出

                # 控制旋转速度
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.linear.y = 0.0
                self.cmd_vel_msg.angular.z = 0.5 * self.rotation_direction # 旋转速度，可以调整
                self.cmd_vel_pub.publish(self.cmd_vel_msg)

            elif self.state=='parking':
                # 在parking状态下，根据激光雷达数据进行停车
                current_distance = 0.0
                with self.scan_lock:
                    if self.scan_data is not None and self.angle_increment is not None:
                        center_idx = len(self.scan_data) // 2
                        scan_range_start = max(0, center_idx - 15) # 假设15个点对应一个范围 (约15度视角)
                        scan_range_end = min(len(self.scan_data), center_idx + 15)
                        
                        valid_distances = []
                        for i in range(scan_range_start, scan_range_end):
                            if self.scan_data[i] > 0.01 and self.scan_data[i] < 5.0: # 过滤掉0和过远的距离
                                valid_distances.append(self.scan_data[i])
                        
                        if len(valid_distances) > 0:
                            current_distance = np.mean(valid_distances) # 计算平均距离
                        else:
                            current_distance = 10.0 # 如果没有有效点，则假设距离很远

                self.distance = current_distance # 更新实例变量

                # PID控制线速度以达到目标距离
                distance_error = self.distance - self.target_distance
                
                self.integral_linear += distance_error * dt
                # 积分项限幅，防止积分饱和
                self.integral_linear = np.clip(self.integral_linear, -0.5, 0.5) 

                derivative_linear = (distance_error - self.last_error_linear) / dt if dt > 0 else 0
                self.last_error_linear = distance_error

                linear_vel_output = self.Kp_linear * distance_error + self.Ki_linear * self.integral_linear + self.Kd_linear * derivative_linear
                
                # 线速度限幅
                linear_vel_output = np.clip(linear_vel_output, -self.max_linear_vel, self.max_linear_vel)

                if abs(distance_error) < 0.1: # **更严格的距离误差阈值 (3cm)**
                    self.cmd_vel_msg.linear.x = 0.0
                    self.cmd_vel_msg.linear.y = 0.0
                    self.cmd_vel_msg.angular.z = 0.0
                    self.cmd_vel_pub.publish(self.cmd_vel_msg)
                    self.state='success'
                    rospy.signal_shutdown("目标物体停车成功，退出")
                else:
                    self.cmd_vel_msg.linear.x = linear_vel_output
                    self.cmd_vel_msg.linear.y = 0.0
                    self.cmd_vel_msg.angular.z = 0.0 # 停车状态下不进行角度和横向调整
                    self.cmd_vel_pub.publish(self.cmd_vel_msg)
                    rospy.loginfo(f"停车中，当前距离: {self.distance:.2f}m, 线速度: {linear_vel_output:.2f}")
                
            # 结束计时并输出推理时间
            inference_end = time.time()
            inference_time = (inference_end - inference_start) * 1000  # 转换为毫秒
            # print(f"Inference time: {inference_time:.2f} ms") # 避免频繁打印

            See_obj=False # 标记是否看到目标物体
            obj_pos=None # 目标物体在图像中的位置 (归一化后的中心x坐标)
            detected_box = None # 存储检测到的目标物体框

            if boxes is not None and len(boxes) > 0:
                # 遍历所有检测到的物体
                for box,class_id, score in zip(boxes, classes, scores):
                    if score < CONFIDENCE_THRESHOLD: # 如果置信度低于阈值，则跳过
                        continue
                    if check_obj(class_id): # 检查是否是目标类型物体
                        See_obj=True # 发现目标物体
                        # 计算目标物体在图像中的中心x坐标（归一化到[-0.5, 0.5]）
                        obj_pos = ((box[0] + box[2]) / 2.0 / IMG_SIZE[0]) - 0.5 
                        
                        # 将检测框坐标转换回原始图像尺寸
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        # 假设 co_helper.letter_box 进行了等比例缩放和居中填充
                        # 需要反向计算原始图像上的坐标
                        scale_x = orig_width / IMG_SIZE[0]
                        scale_y = orig_height / IMG_SIZE[1]
                        
                        # Letterbox的填充计算
                        input_h, input_w = IMG_SIZE[1], IMG_SIZE[0]
                        im_h, im_w = orig_height, orig_width
                        r = min(input_h / im_h, input_w / im_w)
                        new_h, new_w = int(im_h * r), int(im_w * r)
                        pad_h, pad_w = (input_h - new_h) / 2, (input_w - new_w) / 2

                        x1_orig = int((x1 - pad_w) / r)
                        y1_orig = int((y1 - pad_h) / r)
                        x2_orig = int((x2 - pad_w) / r)
                        y2_orig = int((y2 - pad_h) / r)

                        detected_box = [x1_orig, y1_orig, x2_orig, y2_orig, class_id, score]
                        
                        break # 找到目标后即可退出循环

            # --------------------------------------------------------------------
            # 图像标注部分
            if detected_box:
                x1, y1, x2, y2, class_id, score = detected_box
                color = CLASS_COLORS.get(CLASSES[class_id], (0, 255, 255)) # 默认为青色
                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                
                label = f"{CLASSES[class_id]}: {score:.2f}"
                cv2.putText(img_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 绘制目标中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(img_display, (center_x, center_y), 5, (0, 0, 255), -1) # 红色圆点
                
                # 绘制图像中心线
                img_center_x = orig_width // 2
                cv2.line(img_display, (img_center_x, 0), (img_center_x, orig_height), (255, 0, 0), 1) # 蓝色中心线

            # 发布标注后的图像
            annotated_image_msg = self.bridge.cv2_to_imgmsg(img_display, "bgr8")
            self.annotated_pub.publish(annotated_image_msg)
            # --------------------------------------------------------------------

            if See_obj:
                self.state='tracking' # 切换到tracking状态
                rospy.loginfo("进入追踪状态...")

                # 角度误差 (目标在图像中的横向位置)
                # obj_pos 为 [-0.5, 0.5] 范围，0表示居中
                angular_error = obj_pos 
                
                # PID控制角速度
                self.integral_angular += angular_error * dt
                self.integral_angular = np.clip(self.integral_angular, -0.5, 0.5)

                derivative_angular = (angular_error - self.last_error_angular) / dt if dt > 0 else 0
                self.last_error_angular = angular_error

                angular_vel_output = self.Kp_angular * angular_error + self.Ki_angular * self.integral_angular + self.Kd_angular * derivative_angular
                angular_vel_output = np.clip(angular_vel_output, -self.max_angular_vel, self.max_angular_vel)

                # 横向速度PID参数 (使用obj_pos作为误差输入进行侧向调整)
                lateral_error = obj_pos # 使用与角速度相同的视觉横向误差
                self.integral_lateral += lateral_error * dt
                self.integral_lateral = np.clip(self.integral_lateral, -0.2, 0.2) 

                derivative_lateral = (lateral_error - self.last_error_lateral) / dt if dt > 0 else 0
                self.last_error_lateral = lateral_error

                lateral_vel_output = self.Kp_lateral * lateral_error + self.Ki_lateral * self.integral_lateral + self.Kd_lateral * derivative_lateral
                lateral_vel_output = np.clip(lateral_vel_output, -self.max_lateral_vel, self.max_lateral_vel)


                # 评估距离以决定是否进入停车状态
                current_distance_front = 10.0 # 默认一个较大的距离
                with self.scan_lock:
                    if self.scan_data is not None and self.angle_increment is not None:
                        center_idx = len(self.scan_data) // 2
                        scan_range_start = max(0, center_idx - 15)
                        scan_range_end = min(len(self.scan_data), center_idx + 15)
                        
                        valid_distances_front = []
                        for i in range(scan_range_start, scan_range_end):
                            if self.scan_data[i] > 0.01 and self.scan_data[i] < 5.0:
                                valid_distances_front.append(self.scan_data[i])
                        if len(valid_distances_front) > 0:
                            current_distance_front = np.mean(valid_distances_front)
                
                self.distance = current_distance_front # 更新实例变量

                # 根据与目标的距离决定线速度
                if self.distance > self.target_distance + 0.1: # 如果距离目标较远，则前进
                    self.cmd_vel_msg.linear.x = self.max_linear_vel # 简单前进，也可以用PID
                elif self.distance < self.target_distance - 0.05: # 如果距离目标太近，则小幅后退 (0.05m阈值)
                    self.cmd_vel_msg.linear.x = -0.05 # 小幅后退
                else: # 接近目标距离，进行对齐
                    self.cmd_vel_msg.linear.x = 0.0 # 距离合适时，停止前后移动，专注于对齐
                    rospy.loginfo(f"接近目标距离，视觉横向误差: {obj_pos:.3f}, 角度误差: {angular_error:.3f}")
                    # **更严格的对齐阈值，只有当同时满足距离和对齐要求才进入停车**
                    if abs(obj_pos) < 0.02 and abs(angular_error) < 0.02: # **更小阈值，例如0.02 (图像中心的2%)**
                        # 确保也满足距离阈值，避免在对齐过程中突然进入停车
                        if abs(self.distance - self.target_distance) < 0.05: # 距离误差在5cm内
                            self.state='parking'
                            rospy.loginfo("满足停车条件，进入停车状态...")
                    else:
                        # 如果距离合适但对齐不佳，继续进行角度和横向调整
                        rospy.loginfo("距离已合适，正在精细对齐...")


                self.cmd_vel_msg.angular.z = angular_vel_output # 设置角速度
                self.cmd_vel_msg.linear.y = lateral_vel_output # 设置横向速度
                self.cmd_vel_pub.publish(self.cmd_vel_msg)
                rospy.loginfo(f"追踪中 - AngVel: {angular_vel_output:.2f}, LinX: {self.cmd_vel_msg.linear.x:.2f}, LinY: {lateral_vel_output:.2f}, 距离: {self.distance:.2f}")

            else: # 未检测到目标物体
                rospy.loginfo("目标物体未检测到,继续旋转寻找")
                # 在未检测到目标时，保持之前的寻找策略，或可以考虑停止一段时间再搜索
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.linear.y = 0.0
                # 继续以当前方向旋转，这里保持和finding状态一样的旋转速度
                self.cmd_vel_msg.angular.z = 0.5 * self.rotation_direction 
                self.cmd_vel_pub.publish(self.cmd_vel_msg)
            
            
        except Exception as e:
            rospy.logerr(f"Error processing and annotating image: {e}")
            return None
    
    def run(self):
        """
        运行ROS节点，进入自旋状态。
        """
        rospy.spin()
    
    def shutdown(self):
        """
        关闭检测器，释放模型资源。
        """
        if self.model: # 确保模型已初始化
            self.model.release()
        rospy.loginfo("YOLO Continuous Detector shutdown")


if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('yolo_debug', anonymous=False)

    # 从ROS参数服务器获取参数
    # 示例: rosrun your_package your_script.py _model_path:=/path/to/your/model.rknn _target:=rk3588 _device_id:=0
    # 注意: 如果你的ROS包叫 'my_robot_pkg'，脚本叫 'detector_node.py'
    # 运行命令可能是: rosrun my_robot_pkg detector_node.py _model_path:=$(rospack find my_robot_pkg)/models/your_model.rknn
    
    model_path = rospy.get_param('~model_path') # 模型路径
    target = rospy.get_param('~target', 'rk3588') # 目标平台，默认为'rk3588'
    device_id = rospy.get_param('~device_id', 0) # 设备ID，默认为0

    try:
        # 创建YoloContinuousDetector实例
        detector = YoloContinuousDetector(model_path,target, device_id)
        # 设置ROS节点关闭时的回调函数
        rospy.on_shutdown(detector.shutdown)
        # 运行检测器
        detector.run()
    except rospy.ROSInterruptException:
        # 捕获ROS中断异常，通常在Ctrl+C关闭节点时发生
        pass