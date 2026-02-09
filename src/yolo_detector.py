import os
import cv2
import sys
import argparse
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
import threading
import json
import numpy as np
from collections import defaultdict

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


OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.7  # 新增阈值
FRAMES_TO_CAPTURE = 8       # 捕获帧数

CLASSES = ('Cola','Potato','lajiao','Milk','Tomato','Traffic_light_red','Traffic_light_green','Cake','Watermelon','Green_apple','Banana','Traffic_light_no')

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


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


class YoloDetectService:
    def __init__(self, model_path, target='rk3588', device_id=None):
        rospy.init_node('yolo_detect_service', anonymous=True)
        
        # Initialize model
        self.model, self.platform = self.setup_model(model_path, target, device_id)
        
        # Initialize ROS components
        self.bridge = CustomCvBridge()
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        
        # 图像缓存和同步
        self.current_image = None
        self.image_lock = threading.Lock()
        
        # ROS subscribers and service
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.detect_service = rospy.Service('yolo_detect', Trigger, self.detect_service_callback)
        
        rospy.loginfo("YOLO Detect Service initialized")
        rospy.loginfo(f"Model platform: {self.platform}")
        rospy.loginfo("Service 'yolo_detect' is ready")
    
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
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.image_lock:
                self.current_image = cv_image.copy()
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def detect_service_callback(self, req):
        """服务回调函数：检测8帧图像并返回统计结果"""
        response = TriggerResponse()
        
        try:
            rospy.loginfo("Starting 8-frame detection...")
            
            # 收集8帧图像的检测结果
            all_detections = defaultdict(list)  # {class_name: [confidence1, confidence2, ...]}
            
            frames_processed = 0
            
            for frame_idx in range(FRAMES_TO_CAPTURE):
                # 等待新图像
                rospy.sleep(0.1)  # 100ms间隔
                
                with self.image_lock:
                    if self.current_image is None:
                        rospy.logwarn(f"No image available for frame {frame_idx + 1}")
                        continue
                    
                    current_frame = self.current_image.copy()
                
                # 处理当前帧
                boxes, classes, scores = self.process_single_frame(current_frame)
                
                if boxes is not None and len(boxes) > 0:
                    for class_idx, score in zip(classes, scores):
                        class_name = CLASSES[class_idx]
                        all_detections[class_name].append(float(score))
                
                frames_processed += 1
                rospy.loginfo(f"Processed frame {frame_idx + 1}/{FRAMES_TO_CAPTURE}")
            
            # 统计结果：每个类别取最大置信度
            final_results = []
            
            for class_name, confidences in all_detections.items():
                max_confidence = max(confidences)
                
                # 只保留置信度低于0.7的结果
                if max_confidence > CONFIDENCE_THRESHOLD and class_name != "traffic_light_no":
                    final_results.append({
                        "type": class_name,
                        "confidence": f"{max_confidence:.3f}"
                    })
            
            # 生成JSON结果
            result_json = json.dumps(final_results, ensure_ascii=False)
            
            response.success = True
            response.message = result_json
            
            rospy.loginfo(f"Detection completed. Processed {frames_processed} frames.")
            rospy.loginfo(f"Results: {result_json}")
            
        except Exception as e:
            rospy.logerr(f"Service error: {e}")
            response.success = False
            response.message = f"Error: {str(e)}"
        
        return response
    
    def process_single_frame(self, img_src):
        """处理单帧图像"""
        try:
            # Preprocess image
            pad_color = (0, 0, 0)
            img = self.co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=pad_color)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Prepare input data based on platform
            if self.platform in ['pytorch', 'onnx']:
                input_data = img.transpose((2, 0, 1))
                input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
                input_data = input_data / 255.
            else:
                input_data = img
            
            # Run inference
            outputs = self.model.run([input_data])
            boxes, classes, scores = post_process(outputs)
            
            return boxes, classes, scores
            
        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")
            return None, None, None
    
    def run(self):
        rospy.spin()
    
    def shutdown(self):
        self.model.release()
        rospy.loginfo("YOLO Detect Service shutdown")


if __name__ == '__main__':
    #rospy.init_node('sayu_detection_node', anonymous=True)
    parser = argparse.ArgumentParser(description='YOLO11 Detection Service')
    parser.add_argument('--model_path', type=str, required=True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=0, help='device id')

    args = parser.parse_args()

    try:
        service = YoloDetectService(args.model_path, args.target, args.device_id)
        rospy.on_shutdown(service.shutdown)
        service.run()
    except rospy.ROSInterruptException:
        pass