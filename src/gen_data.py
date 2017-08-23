#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:00:10 2017

@author: ray
"""

import numpy as np
import os
import json
import scipy.misc as im
import random
import xml.etree.cElementTree as ET


def read_xml_target(xml,border_perc):
    root = ET.parse(xml)
    #text xml only has one object : text
    all_text = root.findall('object')
    size = root.findall('size')
    
    #return size
    width = int(size[0].find('width').text)
    height = int(size[0].find('height').text)
    
    nontext, text, border = [ np.zeros( ( height, width ), dtype = np.float32 ) for k in range( 3 )]
    nontext = 1 - nontext
    for i in all_text:
        x0 = int(i.find('bndbox').find('xmin').text)
        y0 = int(i.find('bndbox').find('ymin').text)
        x1 = int(i.find('bndbox').find('xmax').text)
        y1 = int(i.find('bndbox').find('ymax').text)
        
        b = max( 1, int( (y1-y0) * border_perc + .5 ) )
        nontext[ y0:y1, x0:x1] = 0
        border[ y0:y1, x0:x1 ] = 1
        y0p, y1p = y0 + b, y1 - b
        x0p, x1p = x0 + b, x1 - b
        text[ y0p:y1p, x0p:x1p ] = 1
        border[ y0p:y1p, x0p:x1p ] = 0
    return np.dstack( [ nontext, border, text ] )

"""
@author: yue_wu
"""
def read_json_target(json_file,border_perc):
    slide_lut = json.load( open( json_file ) )
    image_height, image_width = [ slide_lut[key] for key in [ 'image_height', 'image_width'] ]
    nontext, text, border = [ np.zeros( ( image_height, image_width ), dtype = np.float32 ) for k in range( 3 )]
    nontext = 1 - nontext
    for bbox in slide_lut[ 'bounding_box']:
        x0, y0, w, h = bbox
        x1, y1 = x0+w+1, y0+h+1
        b = max( 1, int( h * border_perc + .5 ) )
        nontext[ y0:y1, x0:x1] = 0
        border[ y0:y1, x0:x1 ] = 1
        y0p, y1p = y0 + b, y1 - b
        x0p, x1p = x0 + b, x1 - b
        text[ y0p:y1p, x0p:x1p ] = 1
        border[ y0p:y1p, x0p:x1p ] = 0
    return np.dstack( [ nontext, border, text ] )








def generator_data(data_list,batch_size=2,border_perc=0.16,resize=True):
    #the data_list contain the path of img and label(json or xml)
    resize_h = 480
    resize_w = 576
    if resize:
        batch_img = np.zeros(shape=(batch_size,resize_h,resize_w,3))
        batch_target = np.zeros(shape=(batch_size,resize_h,resize_w,3))
    
    else:
        batch_size = 1
    while True:
        with open(data_list,'r') as f:
            info = f.readlines()
            random.shuffle(info)
            count = 0
            for i,path in enumerate(info):
                pwd = os.getcwd()
                #blank,enter can be auto split
                img_path,label_path = path.split()
                
                try:
                    #some jpg can not be read:IOError
                    img = im.imread(os.path.join(pwd,img_path),mode='RGB')
                except:
                    continue
                #print img.shape
                
                if label_path.endswith('.xml'):
                    target = read_xml_target(label_path,border_perc)
                elif label_path.endswith('.json'):
                    target= read_json_target(label_path,border_perc)
                else:
                    continue
                if resize:
                    #the other resize method will produce some pixels have 0
                    img = im.imresize(img,(resize_h,resize_w,),interp='nearest')
                    target = im.imresize(target,(resize_h,resize_w),interp='nearest')
                    #print target.shape
                    img = img / 255. - 0.5
                    if img.mean() < 0:
                        img = -img
                    batch_img[count] = img
                    
                    #imresize will change the pixel value from 1 to 255
                    batch_target[count] = target / 255.
                else:
                    batch_img = np.expand_dims(img,0)
                    batch_target = np.expand_dims(target,0)
                count += 1
                if count == batch_size:
                    break
        yield (batch_img,batch_target.astype('uint8'))
        
        
if __name__ == '__main__':
    #a = generator_data('src/data_list.list',batch_size=8,resize=True)
    #b = next(a)
    f=open('src/idcard_list.list','r')
    a=f.readlines()
    b=read_xml_target(a[0].split()[1],0.16)
    print b.shape
    
        

