import numpy as np 
import cv2

class OneWindow:
    def __init__(self, H_start = None, H_end = None, W_start = None, W_end = None, momentX = None, momentY = None):
        self.H_start = H_start
        self.H_end = H_end
        self.H_window = abs(self.H_start - self.H_end)

        self.W_start = W_start
        self.W_end = W_end
        self.W_window = abs(self.W_start - self.W_end)

        self.momentX = momentX
        self.momentY = momentY



class SlidingWindow:
    def __init__(self, rootX, rootY):
        self.max_H_window = 20 #
        self.min_W_window = 20
        self.windows = [OneWindow(
                H_start= rootY - self.max_H_window-1,
                H_end= rootY-1,
                W_start= rootX - self.min_W_window // 2,
                W_end= rootX + self.min_W_window // 2,
                momentX= rootX,
                momentY= rootY)]

    def find_new_window(self):
        last_window = self.windows[len(self.windows)]
        #
        H_start = (last_window.H_start - last_window.H_window)
        H_end = last_window.H_start
        W_start = last_window.W_start
        W_end = last_window.W_end
        #
        mask = self.binaryImg[H_start: H_end, W_start, W_end]
        M = cv2.moments(mask)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if(M['m00'] != 0): #find moment
            return OneWindow(
                H_start = H_start, #
                H_end = H_end, 
                W_start = W_start,
                W_end = W_end,
                momentX = W_start + cx,
                momentY = H_start + cy)
        return None

    def _nextWindow(self):
        new_window = self.find_new_window()


    def slide(self, img):
        self.binaryImg = img.copy()
        self.binaryImgShape = img.shape[:2]

class Curve:
    def __init__(self):
        self.min_window = 10
        self.min_width_windows = 20


        # stored
        self.cur_M = 0 #current moment
        self.roots = None
        self.binaryImg = None
        self.binaryWidth = 320
        self.binaryHeight = 240
        self.cnts = []
        self.rootIdx = -1
    
    def holder(self, img, cnts):
        self.cnts = cnts
        self.binaryImg = img
        self.binaryWidth = img.shape[1]
        self.binaryHeight = img.shape[0]
        self.draw_binary = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for cnt in self.cnts:
            self.roots.append(cnt[cnt[:, :, 1].argmax()][0])

    def next_root(self):
        if self.rootIdx < len(self.roots):
            self.rootIdx += 1
            return True
        return False
        #dieu kien dung

    def next_windows(self):
        mask = self.binaryImg[self.cur_static_window_height:self.cur_static_window_height - self.min_height_window]
        
    def sliding_windows(self):
        self.cur_M = self.roots[self.rootIdx]

        

    def fit(self, img, cnts):
        
        self.holder(img, cnts)
        # while self.next_root():

        # return self.


