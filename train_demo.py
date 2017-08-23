#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:06:27 2017

@author: ray
"""

from src.gen_data import generator_data
from src.models_th import create_multi_res_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import scipy.misc as im
import numpy as np
from src.gen_data import read_xml_target, read_json_target
from tqdm import tqdm
import os


def cross_entropy( y_true, y_pred ) :
    return -K.sum(y_true * K.log(y_pred), axis=1 )


def preprocess_img(img):
    img = im.imresize(img,(480,576,),interp='nearest')
    #print target.shape
    img = img / 255. - 0.5
    if img.mean() < 0:
        img = -img
    return img


def train(model,data_list,batch_size=2,border_perc=0.16,backend='tf'):
    with open(data_list,'r') as f:
        info = f.readlines()
    epoch = 1000
    steps = len(info) // batch_size
    for i in (xrange(1,epoch+1)):
        print '-'*15 , 'Epoch %d' % i , '-'*15
        for j in tqdm(xrange(steps)):
            if backend == 'tf':
                batch_img = np.zeros(shape=(batch_size,480,576,3))
                batch_label = np.zeros(shape=(batch_size,480,576,3))
            else:
                batch_img = np.zeros(shape=(batch_size,3,480,576))
                batch_label = np.zeros(shape=(batch_size,3,480,576))
            count = 0
            size = batch_size
            if j == steps - 1:
                size = batch_size + len(info) % batch_size
            pwd = os.getcwd()
            for k in range(size):
                img_path,label_path = info[j*batch_size+k].split()
                try:
                    img = im.imread(os.path.join(pwd,img_path),mode='RGB')
                    if label_path.endswith('.xml'):
                        target = read_xml_target(label_path,border_perc)
                    elif label_path.endswith('.json'):
                        target= read_json_target(label_path,border_perc)
                    target = im.imresize(target,(480,576),interp='nearest')
                    if backend == 'tf':
                        batch_img[count] = preprocess_img(img)
                        batch_label[count] = target / 255.
                    else:
                        img = preprocess_img(img)
                        batch_img[count] = img.transpose(2,0,1)
                        batch_label[count] = (target / 255.).transpose(2,0,1)
                    count += 1
                except:
                    continue
            print('batch size is %d' % count)
            #model.train_on_batch(batch_img[:count],batch_label[:count].astype('uint8'))
            if j % 10 == 0:
                print('the test loss is %f' % model.evaluate(batch_img[0:8],batch_label[0:8].astype('uint8')))
            if j % 100 == 0:
                print('save the weight done!!')
                if backend == 'tf':
                    model.save_weights('model_weight_tf.h5')
                else:
                    model.save_weights('model_weight_th.h5')


if __name__ == '__main__':
    model = create_multi_res_model()
    model.compile('adam',cross_entropy)

    train_mode = 'sequence'
    if train_mode == 'sequence':
        model.load_weights('model_weight/multi_res.h5')
        train(model,'src/idcard_list.list',12,backend='th')

    
    else:
        sgd = SGD(lr=0.001,momentum=0.9,decay=1e-8,nesterov=True)
        
        gen = generator_data('src/idcard_list.list',batch_size=10)
        modelcheckpoint = ModelCheckpoint('model_weight_th.h5',save_best_only=True,
                                          save_weights_only=True)
        model.fit_generator(gen,samples_per_epoch=10,
                            nb_epoch=5000,
                            callbacks=[modelcheckpoint],
                            validation_data=gen,
                            nb_val_samples=10,
                            nb_worker=2,
                            pickle_safe=True
                            )
        
        
    
    
    
