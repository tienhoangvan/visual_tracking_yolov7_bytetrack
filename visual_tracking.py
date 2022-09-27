#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point

import time
import numpy as np
import cv2
import time
from tracker.byte_tracker import BYTETracker
from utils.visualize import plot_tracking
from tracking_utils.timer import Timer
from inference import Detect


def visual_tracking():
    global pub_human_pose, pub_center_img

    # Detected
    conf_thres = 0.25
    iou_thres = 0.25
    img_size = 640
    weights = "weights/yolov7-tiny.pt"
    device = 0
    half_precision = True
    deteted = Detect(weights, device, img_size, conf_thres, iou_thres,
                     single_cls=False, half_precision=half_precision, trace=False)

    # Tracking
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    frame_rate = 25
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20_check = False

    #print(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    tracker = BYTETracker(track_thresh, track_buffer,
                          match_thresh, mot20_check, frame_rate)
    timer = Timer()
    cap = cv2.VideoCapture(0)

    frame_id = 0
    center = Point()
    human_pose = Point()
    while True:
        _, im0 = cap.read()
        (H, W) = im0.shape[:2]

        frame_id += 1
        if _:
            height, width, _ = im0.shape
            t1 = time.time()
            dets = deteted.detecte(im0)
            online_targets = tracker.update(
                np.array(dets), [height, width], (height, width))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            t2 = time.time()
            timer.toc()
            
            ###### Draw target and publish center of image ######################
            # Center coordinates
            x_center = int(W/2)
            y_center = int(H/2)
            center_coordinates = (x_center, y_center)

            # Radius of circle
            radius_1 = 10
            radius_2 = 20

            # Blue color in BGR
            color = (0, 0, 255)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.circle() method
            # Draw a circle with blue line borders of thickness of 2 px
            im0 = cv2.circle(im0, center_coordinates, radius_1, color, thickness)
            im0 = cv2.circle(im0, center_coordinates, radius_2, color, thickness)
            im0 = cv2.line(im0, (x_center + 5, y_center), (x_center + 15, y_center), color, thickness)
            im0 = cv2.line(im0, (x_center - 5, y_center), (x_center - 15, y_center), color, thickness)
            im0 = cv2.line(im0, (x_center, y_center + 5), (x_center, y_center + 15), color, thickness)
            im0 = cv2.line(im0, (x_center, y_center - 5), (x_center, y_center - 15), color, thickness)

            # publish center of image
            center.x = x_center
            center.y = y_center

            pub_center_img.publish(center)
            print("Published the Center of Image.")

            # grab target pose
            if len(online_tlwhs) > 0:
                x_t, y_t, w_t, h_t = online_tlwhs[0]
                human_pose.x = int(x_t + w_t/2)
                human_pose.y = int(y_t + h_t/2)
                
                pub_human_pose.publish(human_pose)
                print("Published human_pose.")
            else:
                print("human_pose is empty.")

            online_im = plot_tracking(im0, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / 1 / (t2-t1))        
            
            cv2.imshow("Frame", online_im)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            break


if __name__ == '__main__':
    global pub_human_pose, pub_center_img

    try:
        rospy.init_node('visual_tracking')
        pub_human_pose = rospy.Publisher('milbot/human_pose', Point, queue_size=10)
        pub_center_img = rospy.Publisher('milbot/center_image', Point, queue_size=10)

        idle = rospy.Rate(50)
        while not rospy.is_shutdown():
            try:
                visual_tracking()
                idle.sleep()
            except:
                pass

    except rospy.ROSInterruptException:
        pass
