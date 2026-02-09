#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import argparse
import rospy
from sensor_msgs.msg import Image
import threading
import json
import numpy as np
from collections import defaultdict
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

from py_utils.coco_utils import COCO_test_helper


class CustomCvBridge:
    """自定义的CvBridge实现，避免torch库冲突"""
    
    def imgmsg_to_cv2(self, img_msg, desired_encoding="bgr8"):
        """将ROS Image消息转换为OpenCV图像"""
        # 获取图像数据
        if img_msg.encoding == "rgb8":
            # RGB格式，3通道，每像素3字节
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            cv_image = np_arr.reshape((img_msg.height, img_msg.width, 3))
            if desired_encoding == "bgr8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == "bgr8":
            # BGR格式，3通道，每像素3字节
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            cv_image = np_arr.reshape((img_msg.height, img_msg.width, 3))
            if desired_encoding == "rgb8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif img_msg.encoding == "mono8":
            # 灰度图，1通道，每像素1字节
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            cv_image = np_arr.reshape((img_msg.height, img_msg.width))
            if desired_encoding == "bgr8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        elif img_msg.encoding == "16UC1":
            # 16位无符号整数，1通道
            np_arr = np.frombuffer(img_msg.data, np.uint16)
            cv_image = np_arr.reshape((img_msg.height, img_msg.width))
        else:
            raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
        
        return cv_image
    
    def cv2_to_imgmsg(self, cv_image, encoding="bgr8"):
        """将OpenCV图像转换为ROS Image消息"""
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        img_msg.step = cv_image.shape[1] * cv_image.shape[2] if len(cv_image.shape) == 3 else cv_image.shape[1]
        img_msg.data = cv_image.tobytes()
        return img_msg


OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5  # 显示阈值

CLASSES = ('banana','cake','cola','apple','lajiao','milk','potato','tomato','greenlight','redlight','watermelon')

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 类别颜色映射
CLASS_COLORS = {
    'banana': (255, 255, 0),
    'cake': (255, 128, 0),
    'cola': (0, 255, 0),
    'apple': (0, 128, 255),
    'lajiao': (0, 0, 255),
    'milk': (255, 255, 255),
    'potato': (255, 0, 0),
    'tomato': (255, 0, 255),
    'greenlight': (128, 255, 0),
    'redlight': (0, 255, 255),
    'watermelon': (128, 0, 255)
}


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes."""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


class YoloContinuousDetector:
    def __init__(self, model_path, target='rk3588', device_id=None):
        
                
        # Initialize model
        self.model, self.platform = self.setup_model(model_path, target, device_id)
        
        # Initialize ROS components
        self.bridge = CustomCvBridge()
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        
        # 图像缓存和同步
        self.image_lock = threading.Lock()
        self.processing = False
        
        # ROS subscribers and publishers
        self.image_sub = rospy.Subscriber('/image_rect_color', Image, self.image_callback)
        self.annotated_pub = rospy.Publisher('/yolo_detection/annotated_image', Image, queue_size=1)
        
        rospy.loginfo("YOLO Continuous Detector initialized")
        rospy.loginfo(f"Model platform: {self.platform}")
        rospy.loginfo("Publishing annotated images to '/yolo_detection/annotated_image'")
    
    def setup_model(self, model_path, target, device_id):
        if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
            platform = 'pytorch'
            from py_utils.pytorch_executor import Torch_model_container
            model = Torch_model_container(model_path)
        elif model_path.endswith('.rknn'):
            platform = 'rknn'
            from py_utils.rknn_executor import RKNN_model_container 
            model = RKNN_model_container(model_path, target, device_id)
        elif model_path.endswith('onnx'):
            platform = 'onnx'
            from py_utils.onnx_executor import ONNX_model_container
            model = ONNX_model_container(model_path)
        else:
            assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
        return model, platform
    
    def image_callback(self, msg):
        # 避免重复处理
        if self.processing:
            return
            
        self.processing = True
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 处理图像并发布结果
            annotated_image = self.process_and_annotate_image(cv_image)
            
            if annotated_image is not None:
                # Convert back to ROS image and publish
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
                
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
            
            # Prepare input data based on platform
            if self.platform in ['pytorch', 'onnx']:
                input_data = img_rgb.transpose((2, 0, 1))
                input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
                input_data = input_data / 255.
            else:
                input_data = img_rgb
            
            # 开始计时
            inference_start = time.time()
            
            # Run inference
            outputs = self.model.run([input_data])
            boxes, classes, scores = post_process(outputs)
            print(boxes, classes, scores)
            
            # 结束计时并输出
            inference_end = time.time()
            inference_time = (inference_end - inference_start) * 1000  # 转换为毫秒
            print(f"Inference time: {inference_time:.2f} ms")
            
            # 在原始图像上绘制检测结果
            annotated_image = img_src.copy()
            
            if boxes is not None and len(boxes) > 0:
                # 计算缩放比例以将检测框映射回原始图像
                # letter_box后的图像尺寸转换回原始尺寸
                scale_x = orig_width / IMG_SIZE[0]
                scale_y = orig_height / IMG_SIZE[1]
                
                for i, (box, class_idx, score) in enumerate(zip(boxes, classes, scores)):
                    if score > CONFIDENCE_THRESHOLD:
                        # 将检测框坐标转换回原始图像坐标
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, orig_width))
                        y1 = max(0, min(y1, orig_height))
                        x2 = max(0, min(x2, orig_width))
                        y2 = max(0, min(y2, orig_height))
                        
                        class_name = CLASSES[class_idx]
                        color = CLASS_COLORS.get(class_name, (255, 255, 255))
                        
                        # 绘制边界框
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # 绘制标签和置信度
                        label = f"{class_name}: {score:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # 绘制标签背景
                        cv2.rectangle(annotated_image, 
                                    (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        
                        # 绘制标签文字
                        cv2.putText(annotated_image, label, 
                                  (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (0, 0, 0), 2)
            
            return annotated_image
            
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
    model_path = rospy.get_param('~model_path')
    target = rospy.get_param('~target', 'rk3588')
    device_id = rospy.get_param('~device_id', 0)

    try:
        detector = YoloContinuousDetector(model_path,target, device_id)
        rospy.on_shutdown(detector.shutdown)
        detector.run()
    except rospy.ROSInterruptException:
        pass