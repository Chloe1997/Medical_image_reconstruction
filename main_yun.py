import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
from PIL import Image, ImageDraw
import cv2
from skimage.color import rgb2gray
# read img


def downsample(image):
    img_half = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    return img_half

def upsample(image):
    img_half = cv2.resize(image, (0, 0), fx=4, fy=4)
    img_half = cv2.resize(image, (0, 0), fx=4, fy=4)



image_template = cv2.imread('temp.jpg')
image_template = downsample(image_template)
image_org = cv2.imread('target1.jpg')
image_org = downsample(image_org)
org = cv2.imread('target1.jpg')
org = downsample(org)

print(image_template.shape,org.shape)
print(image_template.shape)
print(image_org.shape)

image_org.flags.writeable = True
image_template.flags.writeable = True


if len(image_org.shape) == 3:
    gray = rgb2gray(image_org)
    template1 = rgb2gray(image_template)


def candicate_position(diatance,window_center):
    position = np.array([[window_center[0]-diatance,window_center[1]-diatance],[window_center[0]-diatance,window_center[1]],[window_center[0]-diatance,window_center[1]+diatance],
                          [window_center[0],window_center[1]-diatance],[window_center[0],window_center[1]],[window_center[0],window_center[1]+diatance],
                          [window_center[0]+diatance,window_center[1]-diatance],[window_center[0]-diatance,window_center[1]],[window_center[0]-diatance,window_center[1]+diatance]])

    return position

def position_image(template,point_i,point_j):
    i,j = np.shape(template)
    size_x = [point_i - int(i / 2), point_i + 1+ int(i / 2)]
    size_y = [point_j - int(j / 2), point_j + 1 + int(j / 2)]
    return size_x,size_y

def SSD(template,image,position):
    i,j = np.shape(template)
    # print(position)
    point_i = position[0]
    point_j = position[1]
    size_x, size_y = position_image(template,point_i,point_j)
    # print(size_x,size_y)
    x=0
    y=0
    ssd =0
    for row1 in range(int(size_x[0]),int(size_x[1])) :
        if i>x:
            for column1 in range(int(size_y[0]),int(size_y[1])) :
                if j>y :
                    ssd = ssd + np.square(image[row1,column1]-template[x,y])
                y = y + 1
        x = x + 1
    return ssd

def three_step_search_SSD(window_center,template):
    # print(window_center)
    # First Step
    position1 = candicate_position(4,window_center)
    # print(position)
    ssd = [0 for i in range(9)]
    for i in range(9):
        ssd[i] = SSD(template,gray,position1[i])
    # print(ssd)
    position_center1 = position1[np.argmin(ssd)]
    # print(position_center1)

    # Second Step
    position2 = candicate_position(2,position_center1)
    # print(position)
    ssd = [0 for i in range(9)]
    for i in range(9):
        ssd[i] = SSD(template,gray,position2[i])
    # print(ssd)
    position_center2 = position2[np.argmin(ssd)]
    # print(position_center2)

    # Third Step
    position3 = candicate_position(1,position_center2)
    # print(position)
    ssd = [0 for i in range(9)]
    for i in range(9):
        ssd[i] = SSD(template,gray,position3[i])
    # print(ssd)
    position_center3 = position3[np.argmin(ssd)]
    # print(position_center3)
    min_ssd = min(ssd)
    min_position = position_center3
    return min_position,min_ssd

def graylevel(image):
    histogram = [0 for i in range(0,256)]
    i,j = np.shape(image)
    for row in range(i):
        for column in range(j):
            id = int(image[row,column])
            histogram[id] = histogram[id] + 1
    return histogram
# print(histogram_template)

def PDF(template,image,position):
    i,j = np.shape(template)
    # print(position)
    point_i = position[0]
    point_j = position[1]
    size_x, size_y = position_image(template,point_i,point_j)
    # print(size_x,size_y)
    histogram = [0 for i in range(0,256)]
    for row1 in range(int(size_x[0]),int(size_x[1])) :
        for column1 in range(int(size_y[0]),int(size_y[1])) :
                id = int(image[row1, column1])
                histogram[id] = histogram[id] + 1
    return histogram
def three_step_search_PDF(window_center,template):
    # First Step
    position1 = candicate_position(4,window_center)
    # print(position)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template,gray,position1[i])
        histogram_template = graylevel(template)

        histogram_gray_val = np.dot(np.array(histogram_gray),np.array(histogram_template).T)**0.5
        # print(histogram_gray_val)
        Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + histogram_gray_val

        # for m in range(256):
        #     Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i]+(histogram_gray[m]*histogram_template[m])**0.5
    position_center1 = position1[np.argmax(Bhattacharyya_coefficient)]
    # print(position_center1)

    # Second Step
    position2 = candicate_position(2,position_center1)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template, gray, position2[i])
        histogram_template = graylevel(template)

        histogram_gray_val = np.dot(np.array(histogram_gray),np.array(histogram_template).T)**0.5
        Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + histogram_gray_val

        # for m in range(256):
        #     Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + (
        #                 histogram_gray[m] * histogram_template[m])  0.5
    position_center2 = position2[np.argmax(Bhattacharyya_coefficient)]

    # Third Step
    position3 = candicate_position(1,position_center2)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template, gray, position3[i])
        histogram_template = graylevel(template)

        histogram_gray_val = np.dot(np.array(histogram_gray),np.array(histogram_template).T)**0.5
        Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + histogram_gray_val

        # for m in range(256):
        #     Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + (
        #             histogram_gray[m] * histogram_template[m]) ** 0.5
    position_center3 = position3[np.argmax(Bhattacharyya_coefficient)]
    # print(position_center3)
    max_coefficient = max(Bhattacharyya_coefficient)
    max_position = position_center3
    return max_coefficient,max_position

def three_step_search_PDF(window_center,template):
    # First Step
    position1 = candicate_position(4,window_center)
    # print(position)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template,gray,position1[i])
        histogram_template = graylevel(template)
        for m in range(256):
            Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i]+(histogram_gray[m]*histogram_template[m])**0.5
    position_center1 = position1[np.argmax(Bhattacharyya_coefficient)]
    # print(position_center1)

    # Second Step
    position2 = candicate_position(2,position_center1)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template, gray, position2[i])
        histogram_template = graylevel(template)
        for m in range(256):
            Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + (
                        histogram_gray[m] * histogram_template[m]) ** 0.5
    position_center2 = position2[np.argmax(Bhattacharyya_coefficient)]

    # Third Step
    position3 = candicate_position(1,position_center2)
    Bhattacharyya_coefficient = [0 for i in range(9)]
    for i in range(9):
        histogram_gray = PDF(template, gray, position3[i])
        histogram_template = graylevel(template)
        for m in range(256):
            Bhattacharyya_coefficient[i] = Bhattacharyya_coefficient[i] + (
                    histogram_gray[m] * histogram_template[m]) ** 0.5
    position_center3 = position3[np.argmax(Bhattacharyya_coefficient)]
    # print(position_center3)
    max_coefficient = max(Bhattacharyya_coefficient)
    max_position = position_center3
    return max_coefficient,max_position

def window_search(image,template,is_PDF=True,is_SSD=False):
    t_m,t_n = np.shape(template)
    i_m,i_n = np.shape(image)
    min_ssd, min_position, max_coefficient, max_position = 0, 0, 0, 0

    for row in range(int(t_m/2),int(i_m-t_m/2)-10,50):
        if row < int(i_m-t_m/2)-10 :
            for column in range(int(t_n/2),int(i_n-t_n/2)-10,50):
                if column < int(i_n-t_n/2)-10:
                    if is_SSD == True:
                        # print(row,column)
                        posion,ssd = three_step_search_SSD(window_center=[row,column],template=template)
                        # print(ssd)
                        if ssd < min_ssd or min_ssd == 0 :
                            min_ssd = ssd
                            min_position = posion

                    if is_PDF == True:
                        coefficient, position = three_step_search_PDF(window_center=[row,column],template=template)
                        if coefficient > max_coefficient or max_coefficient == 0 :
                            max_coefficient = coefficient
                            max_position = position
    return min_ssd,min_position,max_coefficient,max_position

# SSD + TSS
import time
end = time.time()
ssd,position,void1,void2 = window_search(gray,template1,is_SSD=True)
rectangle_x, rectangle_y = position_image(template1,position[0],position[1])
print(rectangle_x,rectangle_y)
cv2.rectangle(org, (int(rectangle_y[0]),int(rectangle_x[0])), (int(rectangle_y[1]),int(rectangle_x[1])), (0, 255, 0), 2)
print(time.time()-end)
cv2.imshow('My Image', org)
cv2.waitKey(0)
cv2.destroyAllWindows()


# void1, void2, max_coefficient, max_position = window_search(gray, template1, is_PDF=True)
# rectangle_x, rectangle_y = position_image(template1,max_position[0],max_position[1])
# print(rectangle_x,rectangle_y)
# cv2.rectangle(org, (int(rectangle_y[0]),int(rectangle_x[0])), (int(rectangle_y[1]),int(rectangle_x[1])), (0, 255, 0), 2)
# print(time.time()-end)
# cv2.imshow('My Image', org)
# cv2.waitKey(0)
# cv2.destroyAllWindows()