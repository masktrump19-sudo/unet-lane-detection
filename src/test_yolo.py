#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger
import json

def test_yolo_service():
    rospy.init_node('yolo_service_client', anonymous=True)
    
    # 等待服务可用
    rospy.wait_for_service('yolo_detect')
    
    try:
        # 创建服务代理
        detect_service = rospy.ServiceProxy('yolo_detect', Trigger)
        
        print("Calling YOLO detection service...")
        
        # 调用服务
        response = detect_service()
        
        if response.success:
            print("Detection successful!")
            print("Results:")
            
            # 解析JSON结果
            try:
                results = json.loads(response.message)
                if results:
                    for detection in results:
                        print(f"  - Type: {detection['type']}, Confidence: {detection['confidence']}")
                else:
                    print("  No objects detected with confidence < 0.7")
            except json.JSONDecodeError:
                print(f"  Raw response: {response.message}")
        else:
            print(f"Detection failed: {response.message}")
            
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

if __name__ == '__main__':
    test_yolo_service()