#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber('/drone/img_dep', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            try:    
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except CvBridgeError as e:
                print(e)
            # Get image dimensions
            height, width = cv_image.shape
            # scaled_depth_image   = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX)
            img = np.zeros((height, width))
            # Display the image with pixel values as text
            # Add pixel values as text to the image
            for i in range(0, cv_image.shape[0], 50):
                for j in range(0, cv_image.shape[1], 50):
                    cv2.putText(img, f"{cv_image[i, j]:.2f}", (j, i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Image with Pixel Values', img)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        image_subscriber = ImageSubscriber()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass