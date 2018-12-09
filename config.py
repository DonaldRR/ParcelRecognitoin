from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from utils import *
from config import *

"""
PATH
"""
EAST_MODEL_PATH = '/home/donald/PycharmProjects/Atos/opencv-text-recognition/frozen_east_text_detection.pb'

ROTATED_IMG_DIR = './data/test_rotated_img_box/'
SPLIT_IMG_DIR = './data/test_split_img/'
ORIGIN_IMG_PATH = './data/test_img'
TRAINING_CSV_PATH = './data/csv/combined-training.csv'
X_SAVE_PATH = './X.npy'
y_SAVE_PATH = './y.npy'
CHAR2VEC_PATH = './char_embedding/glove.840B.300d-char.txt'
TRAIN_IMG_PATH = './data/train_img'
WORD2VEC_PATH = './word_embedding/GoogleNews-vectors-negative300.bin'
"""
TRAIN
"""
BATCH_SIZE = 64
EPOCHS = 15