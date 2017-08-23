#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:06:27 2017

@author: ray
"""

from gen_data import generator_data
from models import create_multi_res_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint


def cross_entropy( true_dist, coding_dist ) :
    return -K.sum(true_dist * K.log(coding_dist), axis=-1 )


if __name__ == '__main__':
    model = create_multi_res_model()
    model.compile('adam',cross_entropy)
    gen = generator_data('src/data_list.list')
    modelcheckpoint = ModelCheckpoint('model_weight/',save_best_only=True,
                                      save_weights_only=True)
    '''
    model.fit_generator(gen,samples_per_epoch=16,
                        nb_epoch=10000,
                        callbacks=[modelcheckpoint],
                        validation_data=gen,
                        nb_val_samples=8,
                        nb_worker=2,
                        pickle_safe=True
                        )
    '''
    
