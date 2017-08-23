# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:35:27 2017

@author: Ray
"""

from src.models_th import create_multi_res_model
from src.ppt_text_detector import from_res_map_to_bbox
import scipy.misc as im
import numpy as np
import cv2
import time
import pylab as plt

def preprocess_img(img):
    img = im.imresize(img,(480,576,),interp='nearest')
    #print target.shape
    img = img / 255. - 0.5
    if img.mean() < 0:
        img = -img
    img = img.transpose(2,0,1)
    img = np.expand_dims(img,0)
    return img.astype('float32')

if __name__ =='__main__':
    model = create_multi_res_model()
    model.load_weights('model_weight_th.h5')
    img = im.imread('1.jpg')
    tensor = preprocess_img(img)
    
    
    time_begin = time.time()
    output = model.predict(tensor)
    time_end = time.time()
    print('model_predit takes %f s' % (time_end-time_begin))
    
    res_map = (output[0].transpose(1,2,0))
    plt.imshow(res_map)
    
    #A dict that contain bounding_box and proba
    time_begin = time.time()
    predict = from_res_map_to_bbox(res_map)
    time_end = time.time()
    print('bbox_predit takes %f s' % (time_end-time_begin))
    
    
    img = im.imresize(img,(480,576,),interp='nearest')
    
    
    time_begin = time.time()
    for bbox in predict['bounding_box']:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255),1)
    time_end = time.time()
    print('draw_bbox takes %f s' % (time_end-time_begin))
    
    
    plt.figure(2)
    plt.imshow(img)
    #im.imsave('result.jpg',img)
    
    
