import numpy as np
import cv2
import math
from sensor_msgs.msg import Image

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5  # 显示阈值

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

def fit_line_and_calculate_angle(points):
    """
    拟合二维点集为直线，并计算直线与x轴正半轴的夹角（0到pi）
    
    参数:
        points: 二维点集，格式为[(x1,y1), (x2,y2), ..., (xn,yn)]或np.array
    返回:
        angle: 夹角（弧度，范围0到pi）
        (k, b): 直线斜率和截距
    """
    # 转换为numpy数组
    points = np.array(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    n = len(points)
    
    # 最小二乘法拟合直线 y = kx + b
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    
    # 计算斜率k和截距b
    denominator = n * sum_x2 - sum_x ** 2
    if denominator == 0:
        # 避免垂直直线导致除零（此时斜率无穷大）
        k = np.inf
        b = 0.0
    else:
        k = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y - k * sum_x) / n
    
    # 计算角度（弧度）
    if k == np.inf:
        # 垂直直线（x为常数），与x轴夹角为pi/2
        angle_rad = np.pi / 2
    else:
        # 计算初始角度（范围-pi/2到pi/2）
        angle_rad = math.atan(k)
        # 调整角度到0到pi范围
        if angle_rad < 0:
            angle_rad += np.pi
    
    return angle_rad