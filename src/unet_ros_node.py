#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
import os
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Add path for py_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from py_utils.rknn_executor import RKNN_model_container 

class RKNNLaneInference:
    def __init__(self, rknn_model_path, target='rk3588', device_id='0'):
        self.target = target
        self.device_id = device_id
        
        print('--> Loading RKNN model')
        # 使用RKNN_model_container替代直接的RKNN API
        self.model = RKNN_model_container(rknn_model_path, target, device_id)
        print('RKNN model loaded successfully')
    
    def preprocess_image(self, image):
        """预处理输入图像"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_shape = image.shape[:2]
        
        # 调整大小到模型输入尺寸
        image = cv2.resize(image, (224, 224))
        
        # 针对i8量化模型，保持uint8格式，模型内部已配置归一化
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # 添加batch维度：(H,W,C) -> (1,H,W,C)
        image = np.expand_dims(image, axis=0)
        
        return image, original_shape
    
    def postprocess_output(self, output, original_shape, threshold=0.5):
        """后处理输出结果"""
        # 获取预测结果
        if isinstance(output, list) and len(output) > 0:
            mask = output[0]
        else:
            mask = output
            
        # 如果是4维输出，去除batch维度
        if len(mask.shape) == 4:
            mask = mask[0, 0]  # 假设输出形状为 [1, 1, 224, 224]
        elif len(mask.shape) == 3:
            mask = mask[0]  # 假设输出形状为 [1, 224, 224]
        
        # 处理i8量化模型输出
        if mask.dtype == np.int8:
            mask = mask.astype(np.float32)
        
        # 应用sigmoid激活函数（如果模型输出是logits）
        if mask.max() > 1.0 or mask.min() < 0.0:
            mask = 1 / (1 + np.exp(-mask))  # sigmoid
        
        # 应用阈值
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # 调整回原始尺寸
        binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]))
        
        return binary_mask
    
    def predict(self, image, threshold=0.5):
        """预测单张图像"""
        # 预处理
        input_data, original_shape = self.preprocess_image(image)
        
        # RKNN推理
        start_time = time.time()
        try:
            outputs = self.model.run(inputs=[input_data])
            inference_time = time.time() - start_time
            
            if outputs is None or len(outputs) == 0:
                print("Warning: Model inference returned empty output")
                return np.zeros(original_shape, dtype=np.uint8), inference_time
                
        except Exception as e:
            print(f"Inference error: {e}")
            inference_time = time.time() - start_time
            return np.zeros(original_shape, dtype=np.uint8), inference_time
        
        # 后处理
        mask = self.postprocess_output(outputs, original_shape, threshold)
        
        return mask, inference_time
    
    def predict_video(self, video_path, output_path, threshold=0.5):
        """预测视频"""
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预测车道线
            mask, inference_time = self.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold)
            total_time += inference_time
            
            # 可视化结果
            lane_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            result = cv2.addWeighted(frame, 0.7, lane_colored, 0.3, 0)
            
            # 添加信息文本
            cv2.putText(result, f'RKNN FPS: {1/inference_time:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(result)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, Avg FPS: {frame_count/total_time:.1f}")
        
        cap.release()
        out.release()
        print(f"Video processing completed. Average FPS: {frame_count/total_time:.1f}")
    
    def release(self):
        """释放资源"""
        if hasattr(self, 'model'):
            self.model.release()
            print("RKNN resources released")
    
    def __del__(self):
        """释放资源"""
        self.release()

def benchmark_rknn_model(model_path, test_image_path, num_runs=100):
    """基准测试RKNN模型性能"""
    inferencer = RKNNLaneInference(model_path)
    
    if not os.path.exists(test_image_path):
        print(f"Test image {test_image_path} not found!")
        return
    
    # 预热
    print("Warming up...")
    for _ in range(10):
        inferencer.predict(test_image_path)
    
    # 性能测试
    print(f"Running {num_runs} inference iterations...")
    times = []
    
    for i in range(num_runs):
        _, inference_time = inferencer.predict(test_image_path)
        times.append(inference_time)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_runs} iterations")
    
    # 统计结果
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_fps = 1 / avg_time
    
    print("\n=== Performance Results ===")
    print(f"Average inference time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Min inference time: {min_time:.3f}s")
    print(f"Max inference time: {max_time:.3f}s")
    print(f"Average FPS: {avg_fps:.1f}")

def test_rknn_webcam():
    """测试RKNN模型的摄像头实时推理"""
    model_path = 'models/lane_unet.rknn'
    
    if not os.path.exists(model_path):
        print(f"RKNN model {model_path} not found!")
        return
    
    inferencer = RKNNLaneInference(model_path)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预测
        mask, inference_time = inferencer.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 可视化
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        result = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
        
        # 显示FPS
        fps = 1 / inference_time if inference_time > 0 else 0
        cv2.putText(result, f'RKNN FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('RKNN Lane Detection', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

class LaneSegmentationROS:
    def __init__(self):
        rospy.init_node('lane_segmentation_node', anonymous=True)
        
        self.bridge = CvBridge()
        
        # 定义原图中的四个点（四边形的四个顶点）
        self.src_points = np.float32([
            [29, 347],  # 左下点
            [619, 368], # 右下点
            [202, 238],  # 左上点
            [422, 248]   # 右上点
        ])
        
        # 定义目标图像中的四个点（矩形的四个顶点）
        self.dst_points = np.float32([
            [300, 580],     # 左下点
            [755, 580],     # 右下点
            [300, 100],     # 左上点
            [755, 100]      # 右上点
        ])
        
        # 计算透视变换矩阵
        self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # 输出图像尺寸
        self.output_width = 1055
        self.output_height = 685
        
        # 获取ROS参数
        model_path = rospy.get_param('~model_path', '/home/ucar/ucar_ws/src/rknn_pkg/model/lane_unet_803.rknn')
        threshold = rospy.get_param('~threshold', 0.5)
        target = rospy.get_param('~target', 'rk3588')
        device_id = rospy.get_param('~device_id', '0')
        
        # 初始化RKNN推理器
        try:
            self.inferencer = RKNNLaneInference(model_path, target, device_id)
            self.threshold = threshold
            rospy.loginfo(f"RKNN Lane Segmentation Node initialized with model: {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to initialize RKNN model: {e}")
            raise
        
        # 订阅图像话题
        input_topic = rospy.get_param('~input_topic', '/image_rect_color')
        output_topic = rospy.get_param('~output_topic', '/mask')
        
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback, queue_size=1)
        self.mask_pub = rospy.Publisher(output_topic, Image, queue_size=1)
        
        # 性能统计
        self.frame_count = 0
        self.total_time = 0
        self.last_log_time = time.time()
        
        rospy.loginfo(f"Subscribing to: {input_topic}")
        rospy.loginfo(f"Publishing to: {output_topic}")
        rospy.loginfo("Lane Segmentation with Perspective Transform Node Started")
    
    def image_callback(self, msg):
        try:
            start_time = time.time()
            
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 应用透视变换
            warped_image = cv2.warpPerspective(cv_image, self.perspective_matrix, 
                                             (self.output_width, self.output_height))
            
            # 压缩图像尺寸（保持原始比例）
            compressed_width = int(self.output_width * 1)
            compressed_height = int(self.output_height * 1)
            transformed_image = cv2.resize(warped_image, (compressed_width, compressed_height), 
                                        interpolation=cv2.INTER_AREA)
            
            # 转换为RGB格式（神经网络期望的格式）
            rgb_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            
            # 使用RKNN模型进行车道线分割
            binary_mask, inference_time = self.inferencer.predict(rgb_image, self.threshold)
            
            # 将二值化结果转换为ROS消息
            # binary_mask已经是0-255的uint8格式
            mask_msg = self.bridge.cv2_to_imgmsg(binary_mask, "mono8")
            mask_msg.header = msg.header  # 保持时间戳和frame_id
            
            # 发布二值化结果
            self.mask_pub.publish(mask_msg)
            
            # 性能统计
            total_time = time.time() - start_time
            self.frame_count += 1
            self.total_time += total_time
            
            # 每5秒打印一次性能信息
            current_time = time.time()
            if current_time - self.last_log_time > 5.0:
                avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
                rospy.loginfo(f"Lane Segmentation - Frames: {self.frame_count}, "
                            f"Avg FPS: {avg_fps:.1f}, "
                            f"Last inference: {inference_time:.3f}s")
                self.last_log_time = current_time
                
        except Exception as e:
            rospy.logerr(f"Error in lane segmentation: {e}")
    
    def shutdown_callback(self):
        """节点关闭时的清理函数"""
        rospy.loginfo("Shutting down lane segmentation node...")
        if hasattr(self, 'inferencer'):
            self.inferencer.release()
        rospy.loginfo("Lane segmentation node shutdown complete")
    
    def run(self):
        # 注册关闭回调
        rospy.on_shutdown(self.shutdown_callback)
        rospy.spin()

if __name__ == "__main__":
    # 检查是否在ROS环境中运行
    if 'ROS_MASTER_URI' in os.environ:
        try:
            # ROS模式
            node = LaneSegmentationROS()
            node.run()
        except rospy.ROSInterruptException:
            pass
    else:
        # 原有的测试模式
        print("RKNN Lane Detection Inference")
        print("1. Test single image")
        print("2. Benchmark performance")
        print("3. Test webcam")
        print("4. Test video file")
        
        choice = '4'#input("Choose option (1-4): ")
        
        model_path = '/home/ucar/ucar_ws/src/rknn_pkg/model/lane_unet_final.rknn'
        
        if not os.path.exists(model_path):
            print(f"RKNN model {model_path} not found!")
            print("Please run export_rknn.py first to generate the RKNN model.")
            exit()
        
        if choice == '1':
            test_image = "picture.jpg"#input("Enter test image path: ")
            inferencer = RKNNLaneInference(model_path)
            try:
                mask, inference_time = inferencer.predict(test_image)
                print(f"Inference time: {inference_time:.3f}s, FPS: {1/inference_time:.1f}")
                
                # 显示结果
                original = cv2.imread(test_image)
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                result = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
                
                # cv2.imshow('RKNN Result', result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite('rknn_result.png', mask)
            finally:
                inferencer.release()
        
        elif choice == '2':
            test_image = input("Enter test image path: ")
            benchmark_rknn_model(model_path, test_image)
            
        elif choice == '3':
            test_rknn_webcam()
            
        elif choice == '4':
            video_path = '/home/ucar/ucar_ws/src/rknn_pkg/test_images/video3.mp4'#input("Enter video file path: ")
            output_path = 'video0111.mp4'#input("Enter output video path: ")
            inferencer = RKNNLaneInference(model_path)
            try:
                inferencer.predict_video(video_path, output_path)
            finally:
                inferencer.release()
