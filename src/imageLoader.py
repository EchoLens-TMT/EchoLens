import os
import pandas as pd
import cv2
import numpy as np

class ImageLoader():
    def __init__(self, image_data_path, caption_file):
        self.image_folder = image_data_path
        self.caption_file = caption_file

    def read_images(self):
        img = cv2.imread


img = cv2.imread(r"D:\DIT\First Sem\Computer Vision\EchoLense\DataSet\Images\47871819_db55ac4699.jpg")

cv2.imshow('My Image', img)

cv2.waitKey(0)

cv2.destroyAllWindows()