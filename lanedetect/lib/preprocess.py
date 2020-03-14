import cv2
import numpy as np

IMG_WIDTH = 320
IMG_HEIGHT = 240

class Preprocess:
    def __init__(self):
        self.src_bird_view_point = np.float32([[100,  100],[220,  100],
                                                [0, 180],[320, 180]])

        self.dst_bird_view_point = np.float32([[110,   0], [210,   0],
                                                [100, 240], [220, 240]])

        # self.src_bird_view_point = np.float32([[100,  80],[220,  80],
        #                                         [-30, 195],[350, 195]])

        # self.dst_bird_view_point = np.float32([[100,   0], [220,   0],
        #                                        [100, 240], [220, 240]])

        self.canny_threshold = [150, 300]
        self.gauss_blur_kernel = (3,3)
        self.dilate_kernel = (2,2)

        self.after_birdview_threshold = [10, 255]
        self.after_birdview_erode_kernel = (1, 1)
        self.after_birdview_dilate_kernel = (7, 2)
        self.dotted_edge_height = [20, 60]
        self.min_sliding_windows = 10
        self.debug = 1

    def get_BirdView_Matrix(self):
        return cv2.getPerspectiveTransform(self.src_bird_view_point, self.dst_bird_view_point)

    def get_Inv_BirdView_Matrix(self):
        return cv2.getPerspectiveTransform(self.dst_bird_view_point, self.src_bird_view_point)

    def create_BirdView_Image(self, img):
        return cv2.warpPerspective(img, self.get_BirdView_Matrix(), (320, 240))

    def find_edge(self, img):
        blur = cv2.GaussianBlur(img, self.gauss_blur_kernel, 0)
        canny = cv2.Canny(blur, self.canny_threshold[0], self.canny_threshold[1], None, 3, False)
        dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, self.dilate_kernel), iterations=1)
        return blur, canny, dilate

    def afterBirdView(self, img):
        threshold = cv2.threshold(img, self.after_birdview_threshold[0], self.after_birdview_threshold[1], cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        erosion = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_RECT, self.after_birdview_erode_kernel), iterations=1)
        dilation = cv2.dilate(erosion,cv2.getStructuringElement(cv2.MORPH_RECT, self.after_birdview_dilate_kernel), iterations = 1)
        # dilation = self.extract_vertical_line(dilation)
        return threshold, erosion, dilation

    def showViewPort(self, img):
        drawed = img.copy()

        p1 = (int(self.src_bird_view_point[0][0]),int(self.src_bird_view_point[0][1]))
        p2 = (int(self.src_bird_view_point[1][0]),int(self.src_bird_view_point[1][1]))
        p3 = (int(self.src_bird_view_point[2][0]),int(self.src_bird_view_point[2][1]))
        p4 = (int(self.src_bird_view_point[3][0]),int(self.src_bird_view_point[3][1]))

        cv2.line(drawed, p1, p2, (0,255,255), 2)
        cv2.line(drawed, p2, p4, (0,255,255), 2)
        cv2.line(drawed, p4, p3, (0,255,255), 2)
        cv2.line(drawed, p3, p1, (0,255,255), 2)

        cv2.circle(drawed,p1,2, (0,0,255), 2 )
        cv2.circle(drawed,p2,2, (0,0,255), 2 )
        cv2.circle(drawed,p3,2, (0,0,255), 2 )
        cv2.circle(drawed,p4,2, (0,0,255), 2 )

        cv2.imshow("showViewPort", drawed)
    

#############################################################################
    def extract_vertical_line(self, img, kenel=(3, 2)):
        linek = np.zeros(kenel,dtype=np.uint8)
        linek[...,kenel[1]//2-1]=1
        vertical_line = cv2.morphologyEx(img, cv2.MORPH_OPEN, linek ,iterations=1)
        return vertical_line
    
    def remove_noise(self, img):
        num_sub_border_mask = 1
        mask = np.zeros_like(img)
        mask[num_sub_border_mask:img.shape[0]-num_sub_border_mask, num_sub_border_mask:img.shape[1]-num_sub_border_mask] = 255
        img = cv2.bitwise_and(img, mask)
        contours = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if(area < 50): # change area
                cv2.drawContours(img,[contours[i]],0,(0,0,0),-1)
        return img

    def mask_frame_before_contours(self,preImg):
        blank_image = np.zeros_like(preImg)
        cnts = cv2.findContours(preImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        dotted_cnts = []
        solid_cnts = []
        solid_and_dotted_cnts = []

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # print(area)
            # if area >= 1500 or area < 700:
            minrect = cv2.minAreaRect(cnt)
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
        for dot in dotted_cnts:
            rect = cv2.minAreaRect(dot)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(blank_image,[box],0,(0,255,0), -1)
        for sol in solid_cnts:
            rect = cv2.minAreaRect(sol)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(blank_image,[box],0,(255,255,0),-1)
        
        final = cv2.bitwise_and(blank_image,preImg,mask=blank_image)
        return final

    
    def go(self, img):
        ###############################################
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, _, dilate = self.find_edge(gray)
        dilate = self.remove_noise(dilate)
        birdview = self.create_BirdView_Image(dilate)
        birdview_rgb = self.create_BirdView_Image(img)
        _,_,res = self.afterBirdView(birdview)
        # res = self.mask_frame_before_contours(res)
        if self.debug:
            self.showViewPort(img)
        return birdview_rgb, res
