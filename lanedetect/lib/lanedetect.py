import cv2
import numpy as np 
from lib.preprocess import Preprocess
import lib.util as util
from scipy.optimize import curve_fit

def func(x,a,b):
    return a*np.exp(b/x)

class LaneDetect:
    def __init__(self):
        self.preProcess = Preprocess()
        
        self.debug = 1
        # dotted line detect params
        self.dotted_edge_height = [20, 60]
        self.distance_between_2_dotted_segment = 60
        self.max_dotted_segment = 3
        self.w_frame = 320
        self.h_frame = 240
        # solid line detect params

    def create_mask_line(self, cnts):
        mask = np.zeros((240,320), dtype=np.uint8)
        for cnt in cnts:
            cv2.drawContours(mask,[cnt],0,255,-1)
        return mask

    def fit_line_from_a_contour(self, cnt):
        line = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = line[0]
        vy = line[1]
        x = line[2]
        y = line[3]

        lefty = (-x * vy / vx) + y
        righty = ((320 - x) * vy / vx) + y
        point1 = (320 - 1, righty)
        point2 = (0, lefty)
        return lefty, righty, point1, point2
    
    def get_dotted_solid_contours(self, img):
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        dotted_cnts = []
        solid_cnts = []
        solid_and_dotted_cnts = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # print(area)
            if area >= 1500 or area < 700:
                minrect = cv2.minAreaRect(cnt)
                print(minrect[0][1])
                if minrect[0][1] > 40:
                    box = np.int0(cv2.boxPoints(minrect))
                    edge1 = cv2.norm(box[1] - box[0])
                    edge2 = cv2.norm(box[2] - box[1])
                    longest_edge = edge1 if edge1 > edge2 else edge2
                    if longest_edge > self.dotted_edge_height[1]:
                        solid_cnts.append(cnt)
                        solid_and_dotted_cnts.append(cnt)
                    if longest_edge >= self.dotted_edge_height[0] and longest_edge <= self.dotted_edge_height[1]:
                        dotted_cnts.append(cnt)
                        solid_and_dotted_cnts.append(cnt)
        return dotted_cnts, solid_cnts, solid_and_dotted_cnts


    def find_dotted_line(self, birdview_rgb, dotted_cnts):
        if len(dotted_cnts) < self.max_dotted_segment:
            return None
        
        dotted_holder = []
        for cnt in dotted_cnts:
            extTop = cnt[cnt[:, :, 1].argmin()][0] #[x,y]
            extBot = cnt[cnt[:, :, 1].argmax()][0] #[x,y]
            edge, angle_edge = util.get_angle_longest_edge_from_contour(cnt)

            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            dotted_holder.append([cx, cy, extBot[0], extBot[1], extTop[0], extTop[1], angle_edge])

        dotted_holder = np.array(dotted_holder)
        # dotted_holder = dotted_holder[dotted_holder[:,3].argsort()[::-1]]

        dotted_segments = []
        for i in range(len(dotted_holder)):
            segment = [dotted_holder[i]]
            tail = dotted_holder[i]
            for j in range(i+1, len(dotted_holder)):
                head = dotted_holder[j]
                distance = cv2.norm(np.array([tail[4], tail[5]]) - np.array([head[2], head[3]]))
                if distance <= self.distance_between_2_dotted_segment:
                    segment.append(dotted_holder[j])
                    tail = dotted_holder[j]
            if len(segment) >= self.max_dotted_segment:
                dotted_segments.append(segment)
        
        if len(dotted_segments) < 1:
            return None
        
        max_len_seg = len(dotted_segments[0])
        max_segment = dotted_segments[0]
        for i in range(1, len(dotted_segments)):
            if len(dotted_segments[i]) > max_len_seg:
                max_segment = dotted_segments[i]
                max_len_seg = len(dotted_segments[i])

        x_arr = []
        y_arr = []
        for seg in max_segment:
            x_arr.append(seg[0])
            x_arr.append(seg[2])
            x_arr.append(seg[4])

            y_arr.append(120 + abs(120 - seg[1]) if seg[1] <= 120 else 120 - abs(120 - seg[1]) )
            y_arr.append(120 + abs(120 - seg[3]) if seg[3] <= 120 else 120 - abs(120 - seg[3]) )
            y_arr.append(120 + abs(120 - seg[5]) if seg[5] <= 120 else 120 - abs(120 - seg[5]) )
        
        coeff = np.polyfit(np.array(y_arr), np.array(x_arr), 2)

        ys = np.arange(1, 240, 1)
        # xs = coeff[0] * (ys**2) + coeff[1] * ys + 160
        xs = np.polyval(coeff, ys)
        xs = xs.astype(np.int32)
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            y = 120 + abs(120 - y) if y <= 120 else 120 - abs(120 - y)
            if self.debug:
                cv2.circle(birdview_rgb, (x, y), 2, (0,255,0), 2)


    def find_solid_line(self, solid_cnts):
        for cnt in solid_cnts:
            extTop = cnt[cnt[:, :, 1].argmin()][0] #[x,y]
            extBot = cnt[cnt[:, :, 1].argmax()][0] #[x,y]


    def detect(self, img):
        birdview_rgb, preImg= self.preProcess.go(img)

        # filter solid and dotted lane from binary image
        dotted_cnts, solid_cnts, solid_dotted_cnts = self.get_dotted_solid_contours(preImg)
        solid_dotted_mask = self.create_mask_line(solid_dotted_cnts)
        # extract only dotted lane on a mask
        self.find_dotted_line(birdview_rgb, dotted_cnts)
        self.find_solid_line(solid_cnts)

        # solid_mask = self.create_mask_line(solid_cnts)
        # cv2.imshow("solid mask", solid_mask)
        if self.debug:
            for dot in dotted_cnts:
                rect = cv2.minAreaRect(dot)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(birdview_rgb,[box],0,(0,255,0),1)

            for sol in solid_cnts:
                rect = cv2.minAreaRect(sol)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(birdview_rgb,[box],0,(255,255,0),1)
                # cv2.circle(birdview_rgb,)
            cv2.imshow("solid_dotted_mask", solid_dotted_mask)
            cv2.imshow("pre_img", preImg)
            cv2.imshow("birdview_rgb", birdview_rgb)
        