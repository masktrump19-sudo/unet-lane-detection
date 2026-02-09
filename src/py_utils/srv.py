#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Pose2D

# 导入您的服务定义 (确保包名正确)
from rknn_pkg.srv import DetectTarget, DetectTargetResponse

import threading
import numpy as np
import time

# --- 路径设置 (保持不变) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from py_utils.coco_utils import COCO_test_helper
from tool import CustomCvBridge, post_process

# --- 常量定义 (保持不变) ---
OBJ_THRESH = 0.25  # 对象检测阈值
NMS_THRESH = 0.45  # 非极大值抑制阈值
IMG_SIZE = (640, 640)  # 模型输入图像大小
CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
CLASSES = ('banana', 'cake', 'cola', 'apple', 'lajiao', 'milk', 'potato', 'tomato', 'greenlight', 'redlight', 'watermelon') # 支持的类别
TYPE = ("fruit", "vegetable", "dessert") # 支持的物体类型
CLASS_COLORS = { # 类别对应的颜色，用于可视化
    'banana': (255, 255, 0), 'cake': (255, 128, 0), 'cola': (0, 255, 0),
    'apple': (0, 128, 255), 'lajiao': (0, 0, 255), 'milk': (255, 255, 255),
    'potato': (255, 0, 0), 'tomato': (255, 0, 255), 'greenlight': (128, 255, 0),
    'redlight': (0, 255, 255), 'watermelon': (128, 0, 255)
}


class YoloDetectorService:
    def __init__(self, model_path, target='rk3588', device_id=None):
        # --- 状态和控制变量 ---
        self.state = 'IDLE'  # 机器人当前状态：空闲、寻找、跟踪、泊车、成功、失败
        self.aim_type = None  # 目标物体类型
        self.detected_object_name = ""  # 检测到的物体名称
        self.final_distance = 0.0  # 最终距离目标物体的距离
        
        self.begin_angle = 0.0  # 开始旋转时的角度
        self.counter = 0  # 旋转计数器
        self.rotation_direction = 1  # 旋转方向，1为顺时针，-1为逆时针
        
        self.model, self.platform = self.setup_model(model_path, target, device_id) # 初始化模型
        
        self.bridge = CustomCvBridge()  # OpenCV图像和ROS图像消息转换桥
        self.co_helper = COCO_test_helper(enable_letter_box=True)  # COCO图像处理助手
        
        # --- 关键改变：添加一个变量来保存最新的图像 ---
        self.image_lock = threading.Lock()  # 图像锁，用于同步图像访问
        self.latest_cv_image = None  # 最新的OpenCV图像
        self.last_image_time = rospy.Time(0)  # 最后一次接收到图像的时间

        self.lock = threading.Lock()  # 角度锁，用于同步角度访问
        self.theta = 0.0  # 机器人当前角度
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  # 机器人速度控制发布器
        
        self.scan_lock = threading.Lock()  # 激光雷达数据锁
        self.scan_data = None  # 激光雷达距离数据
        self.intensities = None  # 激光雷达强度数据
        self.angle_increment = None  # 激光雷达角度增量
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)  # 订阅激光雷达数据
        
        self.image_sub = rospy.Subscriber('/image_rect_color', Image, self.image_callback)  # 订阅图像数据
        self.annotated_pub = rospy.Publisher('/yolo_detection/annotated_image', Image, queue_size=1)  # 发布标注后的图像
        self.mag_pose_sub = rospy.Subscriber('/mag_pose_2d', Pose2D, self.mag_pose_callback, queue_size=10)  # 订阅机器人姿态数据

        # --- PID 控制器参数 (完全按照第二个代码设置) ---
        # 角速度PID参数
        self.Kp_angular = 3.5; self.Ki_angular = 0.01; self.Kd_angular = 0.0
        self.integral_angular = 0.0; self.last_error_angular = 0.0
        self.max_angular_vel = 1.0

        # 线速度PID参数 (用于距离控制)
        self.Kp_linear = 0.6; self.Ki_linear = 0.005; self.Kd_linear = 0.0
        self.integral_linear = 0.0; self.last_error_linear = 0.0
        self.target_distance = 0.2; self.max_linear_vel = 0.3

        # 横向速度PID参数 (用于侧向调整)
        self.Kp_lateral = 1.8; self.Ki_lateral = 0.0; self.Kd_lateral = 0.0
        self.integral_lateral = 0.0; self.last_error_lateral = 0.0
        self.max_lateral_vel = 0.1
        
        self.last_time = rospy.Time.now()  # 上次更新时间，用于计算dt

        self.service = rospy.Service('detect_target', DetectTarget, self.handle_detect_target)  # 创建ROS服务

        # --- 新增：用于控制处理线程的成员变量 ---
        self.processing_thread = None
        self.stop_processing_event = threading.Event() # 用于向处理线程发出停止信号

        rospy.loginfo("YOLO 检测服务已初始化。")
        rospy.loginfo(f"模型平台: {self.platform}")
        rospy.loginfo("服务 'detect_target' 已就绪。")

    def handle_detect_target(self, req):
        """
        处理检测目标的服务请求。
        根据请求的目标类型，控制机器人进行寻找、跟踪和泊车。
        """
        rospy.loginfo(f"收到检测请求，目标类型: '{req.target_type}'")

        # 检查是否已有任务正在运行
        if self.processing_thread is not None and self.processing_thread.is_alive():
            rospy.logwarn("检测器正忙于其他任务。请等待当前任务完成或强制停止。")
            return DetectTargetResponse(success=False, detected_object_name="BUSY", final_distance=0.0)

        if req.target_type not in TYPE:
            rospy.logerr(f"无效的目标类型: '{req.target_type}'。有效类型为: {TYPE}")
            return DetectTargetResponse(success=False, detected_object_name=f"INVALID_TYPE: {req.target_type}", final_distance=0.0)

        # --- 初始化任务参数和状态 ---
        self.aim_type = req.target_type  # 设置目标类型
        self.detected_object_name = ""  # 清空检测到的物体名称
        self.final_distance = 0.0  # 清空最终距离
        
        # 重置PID控制器积分和上次误差
        self.integral_angular = 0.0; self.last_error_angular = 0.0
        self.integral_linear = 0.0; self.last_error_linear = 0.0
        self.integral_lateral = 0.0; self.last_error_lateral = 0.0
        
        self.stop_processing_event.clear() # 清除停止信号，允许线程运行
        self.state = 'finding'  # 切换到寻找状态
        with self.lock:
            self.begin_angle = self.theta  # 记录开始寻找时的角度
        self.counter = 0  # 重置旋转计数器
        self.rotation_direction = 1  # 重置旋转方向
        rospy.loginfo("任务开始。进入 '寻找' 状态...")

        # --- 启动独立的处理线程 ---
        self.processing_thread = threading.Thread(target=self._processing_loop_thread)
        self.processing_thread.start()

        # --- 等待处理线程完成任务 ---
        # 服务将在这里阻塞，直到处理线程将状态设置为 'success' 或 'fail'
        # 或者ROS节点被关闭
        self.processing_thread.join() 

        # --- 任务结束，准备响应 ---
        rospy.loginfo(f"任务完成，状态: '{self.state}'")
        
        # 确保机器人停止运动 (如果线程没有自行停止的话)
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        response_success = (self.state == 'success')  # 判断任务是否成功
        response = DetectTargetResponse(
            success=response_success,
            detected_object_name=self.detected_object_name,
            final_distance=self.final_distance if response_success else 0.0
        )
        
        # --- 重置状态以便下次调用 ---
        self.state = 'IDLE'  # 任务完成后回到空闲状态
        self.aim_type = None  # 清空目标类型
        rospy.loginfo("返回 '空闲' 状态。准备接受新的请求。")
        
        return response

    def _processing_loop_thread(self):
        """
        独立的线程函数，负责持续的图像处理和机器人控制逻辑。
        """
        rospy.loginfo("处理线程已启动。")
        while not self.stop_processing_event.is_set() and \
              self.state not in ['success', 'fail'] and \
              not rospy.is_shutdown():
            
            img_to_process = None
            with self.image_lock:
                # 获取最新的图像副本
                if self.latest_cv_image is not None:
                    img_to_process = self.latest_cv_image.copy()

            if img_to_process is not None:
                # 处理图像并根据当前状态控制机器人
                self.process_and_annotate_image(img_to_process)
            else:
                # 如果没有图像，等待一小段时间，避免空转CPU
                rospy.logwarn_throttle(5, "处理线程: 等待第一张图像...")
                time.sleep(0.01) # 10ms sleep

        rospy.loginfo(f"处理线程已停止。最终状态: {self.state}")
        # 线程结束时，确保机器人停止
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)


    def check_obj(self, id):
        """
        根据目标类型检查检测到的物体ID是否符合要求。
        """
        if self.aim_type == 'fruit': return id in [0, 3, 10]  # 香蕉、苹果、西瓜
        elif self.aim_type == 'vegetable': return id in [4, 6, 7]  # 辣椒、土豆、西红柿
        elif self.aim_type == 'dessert': return id in [1, 2, 5]  # 蛋糕、可乐、牛奶
        return False

    def setup_model(self, model_path, target, device_id):
        """
        初始化RKNN模型。
        """
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(model_path, target, device_id)
        return model, 'rknn'
    
    def mag_pose_callback(self, msg):
        """
        ROS姿态消息回调函数，更新机器人当前角度。
        """
        with self.lock:
            self.theta = msg.theta
    
    def scan_callback(self, msg):
        """
        ROS激光雷达消息回调函数，保存激光雷达数据。
        """
        with self.scan_lock:
            self.angle_increment = msg.angle_increment
            self.scan_data = msg.ranges
            self.intensities = msg.intensities
    
    def image_callback(self, msg):
        """
        ROS图像消息回调函数。
        此回调函数现在非常快。它只转换图像并存储起来。
        繁重的处理在独立的线程中完成。
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.latest_cv_image = cv_image
                self.last_image_time = msg.header.stamp
        except Exception as e:
            rospy.logerr(f"图像回调函数出错: {e}")
    
    def process_and_annotate_image(self, img_src):
        """
        处理图像，进行目标检测，并根据当前状态控制机器人。
        """
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        # 确保 dt 大于 0，避免除以零和不必要的更新
        if dt <= 0: 
            # rospy.logwarn_throttle(1, "dt is zero or negative, skipping PID calculation.")
            return 
        self.last_time = current_time

        # 保存原始图像尺寸
        orig_height, orig_width = img_src.shape[:2]
        
        # 图像预处理，进行letterbox缩放
        pad_color = (0, 0, 0) # 填充颜色
        img_resized = self.co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # 转换为RGB格式

        # 运行模型推理
        outputs = self.model.run([img_rgb])
        # 对模型输出进行后处理，得到边界框、类别和分数
        boxes, classes, scores = post_process(outputs)
        
        # 获取用于标注的原始图像副本
        img_display = img_src.copy() 

        # --------------------------------------------------------------------
        # 图像标注部分 (与第二个代码的标注逻辑一致)
        See_obj_in_frame, obj_pos_normalized, detected_name_in_frame, detected_box_orig_coords = \
            self.find_target_in_frame_and_get_coords(boxes, classes, scores, orig_width, orig_height)

        if detected_box_orig_coords:
            x1, y1, x2, y2, class_id, score = detected_box_orig_coords
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

        rospy.loginfo_throttle(1, f"当前状态: {self.state}") # 打印当前机器人状态

        cmd_vel_msg = Twist() # 运动控制消息

        # --- STATE: finding ---
        if self.state == 'finding':
            if See_obj_in_frame:
                rospy.loginfo(f"目标 '{detected_name_in_frame}' 找到! 切换到跟踪状态。")
                self.detected_object_name = detected_name_in_frame
                self.state = 'tracking'
                # 找到目标后，立即停止旋转，让跟踪逻辑接管
                cmd_vel_msg.angular.z = 0.0
                cmd_vel_msg.linear.x = 0.0
                cmd_vel_msg.linear.y = 0.0
                self.cmd_vel_pub.publish(cmd_vel_msg)
                return # 立即返回，等待下一帧在跟踪状态下处理
            else:
                # --- 旋转逻辑 (与第二个代码的寻找逻辑相似) ---
                rotation_angle_threshold = np.pi / 8
                max_rotations = 32 # 增加旋转次数以进行更鲁棒的360度搜索 (32 * 11.25 度)
                with self.lock:
                    # 计算标准化角度差
                    current_angle_diff = (self.theta - self.begin_angle + 2*np.pi) % (2*np.pi)
                
                if current_angle_diff > rotation_angle_threshold: # 检查是否已旋转过一个阈值角度
                    self.counter += 1
                    with self.lock: self.begin_angle = self.theta # 更新起始角度
                    rospy.loginfo(f"寻找状态，已旋转 {self.counter}/{max_rotations} 次")

                if self.counter >= max_rotations:
                    rospy.logwarn("已完成完整旋转，未找到目标。")
                    self.state = 'fail'
                    return # 任务失败，返回

                cmd_vel_msg.angular.z = 0.5 * self.rotation_direction # 旋转速度，可以调整
                self.cmd_vel_pub.publish(cmd_vel_msg)

        # --- STATE: tracking or parking ---
        # 我们将tracking和parking视为一个连续的调整过程，
        # 只是在parking阶段对精度要求更高，且最终停止
        elif self.state in ['tracking', 'parking']:
            if not See_obj_in_frame:
                rospy.logwarn("跟踪过程中目标丢失，继续旋转寻找。")
                # 目标丢失时，停止前进，重新进入寻找状态
                cmd_vel_msg.linear.x = 0.0
                cmd_vel_msg.linear.y = 0.0
                # 让机器人旋转寻找，而不是直接回到finding状态，因为可能只是暂时的遮挡
                cmd_vel_msg.angular.z = 0.5 * self.rotation_direction 
                self.cmd_vel_pub.publish(cmd_vel_msg)
                
                # 如果长时间丢失目标，可以考虑切换回'finding'
                # 或者在这里添加一个计时器，丢失超过X秒则切换回finding
                # 为了简化，这里暂时不加复杂判断
                return 

            # --- PID 和运动逻辑 ---
            # 角度误差 (目标在图像中的横向位置)
            angular_error = obj_pos_normalized # obj_pos_normalized 为 [-0.5, 0.5] 范围，0表示居中
            
            # PID控制角速度
            self.integral_angular += angular_error * dt
            self.integral_angular = np.clip(self.integral_angular, -0.5, 0.5) # 积分项限幅
            derivative_angular = (angular_error - self.last_error_angular) / dt if dt > 0 else 0
            self.last_error_angular = angular_error
            angular_vel_output = self.Kp_angular * angular_error + self.Ki_angular * self.integral_angular + self.Kd_angular * derivative_angular
            cmd_vel_msg.angular.z = np.clip(angular_vel_output, -self.max_angular_vel, self.max_angular_vel)
            
            # 横向速度PID参数
            lateral_error = obj_pos_normalized # 使用与角速度相同的视觉横向误差
            self.integral_lateral += lateral_error * dt
            self.integral_lateral = np.clip(self.integral_lateral, -0.2, 0.2) # 积分项限幅

            derivative_lateral = (lateral_error - self.last_error_lateral) / dt if dt > 0 else 0
            self.last_error_lateral = lateral_error

            lateral_vel_output = self.Kp_lateral * lateral_error + self.Ki_lateral * self.integral_lateral + self.Kd_lateral * derivative_lateral
            cmd_vel_msg.linear.y = np.clip(lateral_vel_output, -self.max_lateral_vel, self.max_lateral_vel)

            # 获取激光雷达距离
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
            distance_error = self.distance - self.target_distance

            # --- 泊车状态的进入条件和成功条件调整 ---
            # 引入更宽松的进入parking状态的阈值
            # 这里的目的是让机器人更早进入“精细调整”阶段，而不仅仅是在很完美的情况下才进入
            # 确保即使进入parking，它也可以继续调整角度和横向
            if abs(distance_error) < 0.2 and abs(obj_pos_normalized) < 0.1 and abs(angular_error) < 0.1: # 距离20cm内，视觉/角度误差10%内
                self.state = 'parking'
                rospy.loginfo_throttle(0.5, "进入泊车精细调整状态...")
            
            # 确定线速度
            if self.state == 'tracking':
                if self.distance > self.target_distance + 0.1: # 如果距离目标较远，则前进
                    cmd_vel_msg.linear.x = self.max_linear_vel 
                elif self.distance < self.target_distance - 0.05: # 如果距离目标太近，则小幅后退
                    cmd_vel_msg.linear.x = -0.05 
                else: # 距离合适时，停止前后移动，专注于对齐
                    cmd_vel_msg.linear.x = 0.0
            elif self.state == 'parking':
                # 在parking状态下，使用PID控制线速度，以便更精确地泊车
                self.integral_linear += distance_error * dt
                self.integral_linear = np.clip(self.integral_linear, -0.5, 0.5) 

                derivative_linear = (distance_error - self.last_error_linear) / dt if dt > 0 else 0
                self.last_error_linear = distance_error

                linear_vel_output = self.Kp_linear * distance_error + self.Ki_linear * self.integral_linear + self.Kd_linear * derivative_linear
                cmd_vel_msg.linear.x = np.clip(linear_vel_output, -self.max_linear_vel, self.max_linear_vel)

                # 泊车成功条件：所有误差都非常小
                # **最终泊车成功条件更严格，需要所有误差都收敛**
                if abs(distance_error) < 0.03 and \
                   abs(obj_pos_normalized) < 0.015 and \
                   abs(angular_error) < 0.015:
                    
                    self.state = 'success' # 切换到成功状态
                    self.final_distance = self.distance # 记录最终距离
                    rospy.loginfo(f"泊车成功，距离 {self.final_distance:.2f}m。物体: {self.detected_object_name}")
                    # 泊车成功后停止机器人
                    stop_msg = Twist()
                    self.cmd_vel_pub.publish(stop_msg)
                    return # 任务成功，返回，结束服务处理循环
                
                rospy.loginfo_throttle(0.5, f"泊车中 - 距离误差: {distance_error:.2f}, 视觉误差: {obj_pos_normalized:.3f}, 角度误差: {angular_error:.3f}")

            self.cmd_vel_pub.publish(cmd_vel_msg)
            rospy.loginfo_throttle(0.5, f"当前速度 - 角速度: {cmd_vel_msg.angular.z:.2f}, 线速度X: {cmd_vel_msg.linear.x:.2f}, 线速度Y: {cmd_vel_msg.linear.y:.2f}, 距离: {self.distance:.2f}")
    
    # 辅助函数，用于获取目标在原始图像中的坐标并进行标注
    def find_target_in_frame_and_get_coords(self, boxes, classes, scores, orig_width, orig_height):
        """
        Helper function to find a valid target, calculate its normalized position,
        and return its original image coordinates for drawing.
        """
        obj_pos = None
        detected_name = None
        detected_box_orig_coords = None

        if boxes is not None:
            for box, class_id, score in zip(boxes, classes, scores):
                if score >= CONFIDENCE_THRESHOLD and self.check_obj(class_id):
                    # Found a valid target
                    obj_pos = ((box[0] + box[2]) / 2.0 / IMG_SIZE[0]) - 0.5 # 归一化到[-0.5, 0.5]
                    detected_name = CLASSES[class_id]

                    # 将检测框坐标转换回原始图像尺寸 (与第二个代码的标注逻辑一致)
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    input_h, input_w = IMG_SIZE[1], IMG_SIZE[0]
                    im_h, im_w = orig_height, orig_width
                    r = min(input_h / im_h, input_w / im_w)
                    new_h, new_w = int(im_h * r), int(im_w * r)
                    pad_h, pad_w = (input_h - new_h) / 2, (input_w - new_w) / 2

                    x1_orig = int((x1 - pad_w) / r)
                    y1_orig = int((y1 - pad_h) / r)
                    x2_orig = int((x2 - pad_w) / r)
                    y2_orig = int((y2 - pad_h) / r)
                    detected_box_orig_coords = [x1_orig, y1_orig, x2_orig, y2_orig, class_id, score]
                    
                    # 返回 True, 归一化位置, 检测到的名称, 原始坐标框
                    return True, obj_pos, detected_name, detected_box_orig_coords
        # No valid target found
        return False, None, None, None


    def run(self):
        """
        启动ROS节点，保持运行。
        """
        rospy.spin()
    
    def shutdown(self):
        """
        关闭模型并记录服务关闭信息。
        """
        # 在关闭前向处理线程发送停止信号并等待其结束
        self.stop_processing_event.set()
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0) # 等待2秒，确保线程有时间停止
            if self.processing_thread.is_alive():
                rospy.logwarn("处理线程未能正常关闭。")

        self.model.release()
        rospy.loginfo("YOLO 检测服务已关闭。")

if __name__ == '__main__':
    rospy.init_node('yolo_service_node', anonymous=False) # 初始化ROS节点

    # 确保在launch文件中设置了model_path参数
    model_path = rospy.get_param('~model_path')
    target = rospy.get_param('~target', 'rk3588')
    device_id = rospy.get_param('~device_id', 0)

    try:
        detector = YoloDetectorService(model_path, target, device_id) # 创建YoloDetectorService实例
        rospy.on_shutdown(detector.shutdown) # 注册关闭回调函数
        detector.run() # 运行服务
    except rospy.ROSInterruptException:
        pass
    except KeyError:
        rospy.logerr("获取 '~model_path' 参数失败。请在您的启动文件中设置它。")