import config
from config import *
import numpy as np
from os.path import join
import os
import cv2
import pandas as pd
import pytesseract
from tqdm import tqdm


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < config.args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


# import the necessary packages
import numpy as np
import cv2


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def rotate(image, pt1, pt2):
    rows = image.shape[0]
    cols = image.shape[1]

    x0, y0 = pt1[0], pt1[1]
    x1, y1 = pt2[0], pt2[1]
    if x1 == x0:
        x1 = x0+1
    incline = abs((y1 - y0) / (x1 - x0))

    if incline < 1:
        if x1 > x0:
            return image
        else:
            # right 180
            rotated_image = cv2.flip(image, -1)
    else:
        if y1 > y0:
            # left 90
            M = cv2.getRotationMatrix2D((cols / 2, cols / 2), 90, 1)
        else:
            # right 90
            M = cv2.getRotationMatrix2D((rows / 2, rows / 2), 270, 1)

        rotated_image = cv2.warpAffine(image, M, (rows, cols))

    return rotated_image


# Get Image and boxes
def getImageBoxes(image_name, image_dir):
    image_path = join(image_dir, image_name) + '.jpg'
    box_path = join(image_dir, image_name) + '.txt'

    # Read Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Read Bounding boxes
    box_file = open(box_path)
    boxes = box_file.readlines()
    boxes = [np.reshape(a=box.replace('\n', '').split(','), newshape=(4, 2)) for box in boxes]
    boxes = np.array(boxes, dtype=int)

    return img, boxes


def rotateOrthog(image_name, image, boxes):
    w_2_hs = [(abs(box[0][0] - box[1][0])) / float((abs(box[0][1] - box[3][1]))) for box in boxes]
    w_2_h = np.average(w_2_hs)

    rows = image.shape[0]
    cols = image.shape[1]

    if w_2_h < 1:
        M = cv2.getRotationMatrix2D((cols / 2, cols / 2), 90, 1)
        rotated_image = cv2.warpAffine(image, M, (rows, cols))

        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                x_ = boxes[i][j][0]
                y_ = boxes[i][j][1]
                boxes[i][j][0] = y_
                boxes[i][j][1] = cols - x_

        return rotated_image, boxes

    return image, boxes


# Paint bounding boxes
def paintBOX(image, boxes):
    for i in range(len(boxes)):
        image = cv2.drawContours(image, [boxes[i]], contourIdx=-1, color=(255, 255, 255), thickness=-1)

    return image


# Expand bounding box
def expandBox(box, ratio):
    w = abs(box[0][0] - box[1][0])
    h = abs(box[0][1] - box[3][1])

    x_min = int(box[0][0] - w * ratio)
    y_min = int(box[0][1] - h * ratio)
    x_max = int(box[2][0] + w * ratio)
    y_max = int(box[2][1] + h * ratio)

    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])


# For number of iteration:
# Find contours
# Get boxes
# Expand boxes
# draw boxes

def generateCandidateBBOX(image, boxes, num_iteration):
    # Binarization
    image, thresh = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

    bbox = []
    #     tmp_images = []

    for i in range(num_iteration):
        # Find Contours
        if len(thresh.shape) == 3:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh, cnts, hierarchy = cv2.findContours(thresh, 1, 2)
        cnts = [np.squeeze(cnt, axis=1) for cnt in cnts]

        # Re-Construct Bounding Boxes
        rects = []
        for cnt in cnts:
            x_min = np.min(cnt[:, 0])
            x_max = np.max(cnt[:, 0])
            y_min = np.min(cnt[:, 1])
            y_max = np.max(cnt[:, 1])
            if (y_max - y_min) * (x_max - x_min) >= 100:
                rects.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
        bbox.extend(rects)

        # Visualize Boxes
        # thresh_b = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        # for rect in rects:
        #     thresh_b = cv2.drawContours(thresh_b, [rect], contourIdx=-1, color=(0,255,0), thickness=-1)

        # Expanding Bounding Boxes
        if i != 0:
            for j in range(len(rects)):
                rects[j] = expandBox(rects[j], 0.05)

        # Draw new Bounding Boxes
        # for rect in rects:
        #     thresh = cv2.drawContours(thresh, [rect], contourIdx=-1, color=(255, 255, 255), thickness=-1)

        # Save
        # tmp_images.append(thresh_b)
        # cv2.imwrite('test_box_{}.jpg'.format(i), thresh_b)
        # cv2.imwrite('test_{}.jpg'.format(i), thresh_)

    return bbox


# # Visualization
# n_col = 4
# n_row = int(np.ceil(n_iter/n_col))

# fig, axes = plt.subplots(n_row,n_col, figsize=(20,20))

# for i in range(n_iter):
#     cur_axe = axes[i//n_col][i%n_col]
#     cur_axe.imshow(tmp_images[i])
def saveSplitImages(image, bbox, save_dir, image_name):
    save_path = join(save_dir, image_name)
    try:
        os.mkdir(save_path)
    except:
        print('# Directory {} already exists.'.format(join(save_path)))

    bbox_dict = {}
    for b in range(len(bbox)):
        cv2.imwrite(save_path + '/box_{}.jpg'.format(b), image[bbox[b][0][1]:bbox[b][3][1],
                                                         bbox[b][0][0]:bbox[b][1][0], :])
        bbox_dict['box_{}.jpg'.format(b)] = bbox[b]
    np.save(save_path+'/bbox.npy', bbox_dict)

def showImageDrawBbox(image, boxes):
    for box in boxes:
        image = cv2.drawContours(image, [box], contourIdx=-1, color=(0, 255, 0), thickness=2)
    plt.imshow(image)