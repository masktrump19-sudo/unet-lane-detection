#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
# 导入服务类型，需要替换为你的实际服务类型包名
from rknn_pkg.srv import DetectTarget,DetectTargetResponse,DetectTargetRequest

def yolo_service_client():
    # 初始化节点
    rospy.init_node('yolo_service_client_node', anonymous=True)
    
    # 等待服务可用，超时10秒
    service_name = '/yolo_tracker'
    rospy.loginfo(f"等待服务 {service_name} 可用...")
    try:
        rospy.wait_for_service(service_name, timeout=10.0)
    except rospy.ROSException:
        
        rospy.logerr(f"服务 {service_name} 不可用，超时退出")
        return
    
    try:
        # 创建服务客户端
        yolo_service = rospy.ServiceProxy(service_name, DetectTarget)
        
        # 创建请求消息
        request = DetectTargetRequest()
        request.aim_type = 'fruit'  # 设置目标类型
        request.if_parking = True  # 设置是否停车
        
        # 发送请求并获取响应
        rospy.loginfo(f"发送请求: aim_type='{request.aim_type}', if_parking={request.if_parking}")
        response = yolo_service(request)
        
        # 打印响应结果
        rospy.loginfo("\n服务响应结果:")
        rospy.loginfo(f"if_success: {response.if_success}")
        rospy.loginfo(f"obj_name: {response.obj_name}")
        
    except rospy.ServiceException as e:
        rospy.logerr(f"服务调用失败: {e}")

if __name__ == "__main__":
    try:
        yolo_service_client()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    