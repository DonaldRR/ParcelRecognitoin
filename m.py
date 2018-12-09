
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import cv2
import os
import sys
import csv
from tqdm import tqdm


"""
Initalize environment
"""


"""
Calculate Box's Direction Vector

calculate a vector which represent the direction of the box
this vector has a slope [-1,1]
"""
def calDirectionVector(box):
    v = np.zeros([4,2])
    v[0]= box[0]-box[1]
    v[1] = box[1]-box[2]
    v[2] = box[2]-box[3]
    v[3] = box[3]-box[0]
    avgVector = np.zeros([2])
    for vector in v:
        nV = normalize(vector.reshape(1,-1))[0]
        if nV[1] == 0:
            nV = np.array([1,0])
        else:
            # calculate slope
            # make sure it's in [-1,1]
            x = nV[0]
            y = nV[1]
            if x<0:
                x = -x
                y = -y
            if y>x:
                nV = (y,-x)
            elif -y > x:
                nV = (-y,x)
            avgVector += nV
    avgVector = avgVector/4
    return avgVector

"""

"""
def getTextDirection(boxes):
    direct = np.zeros([2])
    for box in boxes:
        v = box[2] - box[1]
        v = normalize([v])[0]
        direct += v
    return direct/len(boxes)


"""
calcualte all direction vectors in one image
"""
def calAllDirectionVectors(boxes):
    vectors = np.zeros([boxes.shape[0],2])
    for i in range(boxes.shape[0]):
        box = boxes[i]
        vectors[i] = calDirectionVector(box)
    return vectors


def ifTwoBoxParallel(box1,box2):
    return False

def ifTwoBoxOverlap(box1,box2):
    return False

def mergeTwoBox(box1,box2):
    box = []
    return box


def mergeBoxes(boxes):
    pass


def test_calDirectionVector():

    box = np.array([[2442,1150],[2529,1153],[2528,1179],[2441,1176]])
    direction = calDirectionVector(box)
    image = np.zeros([3000,3000,3])


    cv2.line(image,(1500,1500),tuple(int(i) for i in (1500,1500)+direction*1000),(255,255,255),10)
    cv2.polylines(image,[box],True,(255,255,255),10)

    cv2.namedWindow("d",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("d",(900,900))
    cv2.imshow("d",image)
    cv2.waitKey()

"""
Read data from a dir

yield (image,boxes)
"""
def loadData(boxPath,imgPath):
    for root,dirs,files in os.walk(boxPath):

        for file in files:
            if os.path.splitext(file)[-1] == ".txt":
                boxes = []
                with open(os.path.join(root,file),'r') as f:
                    for line in f.readlines():
                        tokens = line.split(',')
                        box = np.zeros([4,2])
                        for i in range(0,4):
                            box[i] = tokens[i*2:i*2+2]
                        boxes.append(box)
                boxes = np.asarray(boxes)
                image_path = os.path.join(imgPath,os.path.splitext(file)[0]+".jpg")
                image = cv2.imread(image_path)

                yield (image,boxes,os.path.splitext(file)[0])



def showImage(img):
    cv2.imshow("d",img)


def initializeCV():
    cv2.namedWindow("d",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("d",(900,900))

"""
draw all boxes on the image
"""
def drawBoxes(img,boxes):
    cv2.polylines(img,np.int32(boxes),True,(255,0,0))

    return img


def rotateImage(image,boxes,dv):

    if dv[0]<0:
        sign = +1
    else:
        sign = -1
    center = (image.shape[1]/2,image.shape[0]/2)
    # calculate degree
    v = np.array([0,1])
    prod = dv[0]*v[0] +dv[1]*v[1]
    v_len = (v[0]**2 + v[1]**2)**0.5
    dv_len = (dv[0]**2 + dv[1]**2)**0.5
    degree = np.arccos(prod/(v_len*dv_len))
    M = cv2.getRotationMatrix2D(center,  sign*degree*360/(2*3.14159),1)  # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    rotated = cv2.warpAffine(image, M, (image.shape[1],image.shape[0]))
    r_boxes = rotateBoxes(boxes,center,sign*degree)
    return rotated,r_boxes

def rotateBoxes(boxes,center,degree):

    boxes = boxes.reshape(-1,2)
    num = boxes.shape[0]
    # x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
    # y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
    centerM = np.array([center]*num)
    rotateM = np.array([[np.cos(degree),-np.sin(degree)],
                        [np.sin(degree),np.cos(degree)]])
    r_boxes = np.dot(boxes-centerM,rotateM)+centerM
    r_boxes = r_boxes.reshape(-1,4,2)


    return r_boxes




def saveData(img,boxes,dstPath,fileName):
    cv2.imwrite(os.path.join(dstPath,fileName+'.jpg'),img)
    f = open(os.path.join(dstPath,fileName+'.txt'),"w")
    for box in boxes:
        l = list(box.reshape(-1))
        f.write(','.join([str(int(i)) for i in l])+'\n')
    f.close()



def do_rotation(image,boxes):
    vector = getTextDirection(boxes)
    image,boxes = rotateImage(image,boxes,vector)
    return image,boxes


if __name__ == '__main__':

    initializeCV()
    boxPath = "/home/donald/PycharmProjects/Atos/data/test_east_img_bbox"
    imgPath = "/home/donald/PycharmProjects/Atos/data/cutted_data"
    dstPath = "/home/donald/PycharmProjects/Atos/data/test_rotated_img_box"
    csvPath = ""

    for image,boxes,fileName in loadData(boxPath,imgPath):
        vector = getTextDirection(boxes)
        #start = tuple(int(i) for i in (image.shape[1]/2,image.shape[0]/2))
        #end = tuple(int(i) for i in start+vector*1000)
        #cv2.line(image,start,end,(255,255,255))
        try:
            image,boxes = rotateImage(image,boxes,vector)
        except:
            print('Error:{}'.format(fileName))
            continue
        #image = drawBoxes(image,boxes)
        saveData(image,boxes,dstPath,fileName)

        #showImage(image)
        #cv2.waitKey()

    cv2.destroyAllWindows()
