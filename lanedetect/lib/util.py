import cv2
import numpy as np 
from math import acos, sqrt

def get_box_point_from_contour(cnt):
    minrect = cv2.minAreaRect(cnt)
    return np.int0(cv2.boxPoints(minrect))

def get_longest_edge_from_contour(cnt):
    box = get_box_point_from_contour(cnt)
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]

    dis_edge1 = cv2.norm(edge1)
    dis_edge2 = cv2.norm(edge2)

    return edge2 if dis_edge2 < dis_edge1 else edge1


def get_angle_longest_edge_from_contour(cnt): 
    #longest_edge, longest_dis, angle
    box = get_box_point_from_contour(cnt)
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]

    dis_edge1 = cv2.norm(edge1)
    dis_edge2 = cv2.norm(edge2)

    longest_edge = edge1
    longest_dis = dis_edge1

    if longest_dis < dis_edge2:
        longest_dis = dis_edge2
        longest_edge = edge2
    
    hor = np.array([1,0])
    angle = 180/np.pi * acos((hor[0]*longest_edge[0] + hor[1]*longest_edge[1])/ (cv2.norm(hor)*cv2.norm(longest_edge)))

    # angle > 110 and angle < 160: left
    # angle > 20 and angle < 70: right
    return longest_edge, angle
    