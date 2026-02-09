import cv2
import numpy as np
import time

# add path
import sys
import os
# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录的路径（parent_dir）
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到系统路径
sys.path.append(parent_dir)

from py_utils.rknn_executor import RKNN_model_container 


class LaneDetectionRKNN:
    def __init__(self, model_path, num_points=6):
        self.num_points = num_points
        self.input_size = (224, 224)
        
        print('--> Loading RKNN model')
        # 初始化RKNN,加载RKNN模型
        self.model = RKNN_model_container(model_path,'rk3588','0')
        print('done')
        
    
    def preprocess(self, image):
        """
        图像预处理 - 针对i8量化模型的RK3588平台
        RKNN模型已配置了内置的mean/std归一化，输入应为BGR uint8格式
        """
        # 调整图像大小到模型输入尺寸
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # RKNN i8模型期望BGR格式的uint8输入
        # 模型内部已配置mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        # 输入数据范围应为[0, 255]
        input_data = resized.astype(np.uint8)
        
        # RKNN模型需要4维输入：(batch_size, channels, height, width) 或 (batch_size, height, width, channels)
        # 根据错误信息，需要将3维(H,W,C)转换为4维(1,H,W,C)
        input_data = np.expand_dims(input_data, axis=0)  # 添加batch维度
        
        return input_data
    
    def postprocess(self, output, original_shape):
        """
        后处理 - 针对i8量化模型输出的处理
        """
        # 获取原始图像尺寸
        original_height, original_width = original_shape[:2]
        
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")  # 调试信息
        
        # i8量化模型的输出可能需要反量化
        # 输出数据类型可能是int8或float32，需要适当处理
        if output.dtype == np.int8:
            # 如果是int8输出，需要转换为float32并进行反量化
            output = output.astype(np.float32)
            # 根据量化参数进行反量化，这里假设输出已经是归一化坐标
            # 实际使用时可能需要根据具体的量化参数调整
        
        # 如果输出是多维的，需要flatten
        if len(output.shape) > 1:
            output = output.flatten()
        
        # 确保输出长度正确（12个值：6个点×2个坐标）
        if len(output) != self.num_points * 2:
            print(f"Warning: Expected {self.num_points * 2} outputs, got {len(output)}")
            # 如果输出长度不对，用零填充或截断
            if len(output) < self.num_points * 2:
                padded_output = np.zeros(self.num_points * 2)
                padded_output[:len(output)] = output
                output = padded_output
            else:
                output = output[:self.num_points * 2]
        
        # 确保输出在合理范围内 [0, 1]
        output = np.clip(output, 0.0, 1.0)
        
        # 重塑输出为点坐标 (6个点, 每个点2个坐标)
        points = output.reshape(-1, 2)
        
        # 将归一化坐标转换为原始图像坐标
        points[:, 0] *= original_width   # x坐标
        points[:, 1] *= original_height  # y坐标
        
        return points
    
    def visualize_result(self, image, points):
        """
        在图像上可视化检测结果
        """
        result_image = image.copy()
        
        # 绘制点 - 使用更明显的颜色和大小
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255)]
        for i, (x, y) in enumerate(points):
            color = colors[i % len(colors)]
            cv2.circle(result_image, (int(x), int(y)), 3, color, -1)
            cv2.circle(result_image, (int(x), int(y)), 4, (255, 255, 255), 2)
            cv2.putText(result_image, f'{i+1}', (int(x)+15, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 绘制连接线 - 车道中心线
        if len(points) > 1:
            pts = points.astype(np.int32)
            cv2.polylines(result_image, [pts], False, (255, 0, 0), 3)
        
        return result_image
    
    def inference(self, image):
        """
        单张图像推理 - 优化版本用于RK3588 i8模型
        """
        # 记录原始图像形状
        original_shape = image.shape
        
        # 预处理
        input_data = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        try:
            outputs = self.model.run(inputs=[input_data])
            inference_time = time.time() - start_time
            
            # 检查输出是否有效
            if outputs is None or len(outputs) == 0:
                print("Warning: Model inference returned empty output")
                return np.zeros((self.num_points, 2)), inference_time
                
        except Exception as e:
            print(f"Inference error: {e}")
            return np.zeros((self.num_points, 2)), 0.0
        
        # 后处理
        points = self.postprocess(outputs[0], original_shape)
        
        return points, inference_time
    
    def inference_image(self, image_path, output_path=None, show_result=False):
        """
        图像文件推理
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        print(f"Original image shape: {image.shape}")
        
        # 推理
        points, inference_time = self.inference(image)
        
        # 可视化
        result_image = self.visualize_result(image, points)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        # 显示结果
        if show_result:
            # 调整显示窗口大小
            display_image = result_image.copy()
            height, width = display_image.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
            
            cv2.imshow('Lane Detection Result - RK3588', display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"Inference time: {inference_time*1000:.2f}ms")
        print(f"FPS: {1.0/inference_time:.2f}" if inference_time > 0 else "FPS: N/A")
        print("Detected lane centerline points:")
        for i, (x, y) in enumerate(points):
            print(f"Point {i+1}: ({x:.2f}, {y:.2f})")
        
        return points
    
    def inference_video(self, video_path, output_path):
        """
        视频推理 - 针对RK3588优化
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_inference_time = 0
        max_fps = 0
        min_fps = float('inf')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 推理
            points, inference_time = self.inference(frame)
            total_inference_time += inference_time
            
            # 计算FPS统计
            if inference_time > 0:
                current_fps = 1.0 / inference_time
                max_fps = max(max_fps, current_fps)
                min_fps = min(min_fps, current_fps)
            
            # 可视化
            result_frame = self.visualize_result(frame, points)
            
            # 添加性能信息到视频帧
            fps_text = f"FPS: {1.0/inference_time:.1f}" if inference_time > 0 else "FPS: N/A"
            cv2.putText(result_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 写入输出视频
            out.write(result_frame)
            
            # 显示进度
            if frame_count % 30 == 0:  # 每30帧显示一次进度
                progress = (frame_count / total_frames) * 100
                avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), Avg FPS: {avg_fps:.2f}")
        
        # 释放资源
        cap.release()
        out.release()
        
        # 统计信息
        avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
        print(f"\nVideo processing completed!")
        print(f"Output saved to: {output_path}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Max FPS: {max_fps:.2f}")
        print(f"Min FPS: {min_fps:.2f}")
        print(f"Total inference time: {total_inference_time:.2f}s")
    
    def release(self):
        """
        释放RKNN资源
        """
        self.model.release()
        print("RKNN resources released")

def main():
    # 初始化检测器
    model_path = '/home/ucar/ucar_ws/src/rknn_pkg/model/fp.rknn'
    
    print("Initializing Lane Detection on RK3588...")
    detector = LaneDetectionRKNN(model_path)
    
    try:
        # # 图像推理示例
        # image_path = '/home/ucar/ucar_ws/src/rknn_pkg/test_images/frame_001410.jpg'
        # if os.path.exists(image_path):
        #     print("\nRunning image inference...")
        #     detector.inference_image(image_path, 'fp.jpg', show_result=False)
        
        # 视频推理示例
        video_path = '/home/ucar/ucar_ws/src/rknn_pkg/test_images/video.mp4'
        if os.path.exists(video_path):
            print("\nRunning video inference...")
            detector.inference_video(video_path, 'output_video_rk3588.mp4')
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        # 释放资源
        detector.release()
        print("Resources released")

if __name__ == '__main__':
    import os
    main()