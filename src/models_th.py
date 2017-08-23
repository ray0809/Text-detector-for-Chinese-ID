# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:38:32 2017

@author: yue_wu

This script defines required Keras Theano-backend models for PPT text detection
with border support
"""

import keras
#assert keras.__version__ == '1.1.0', "ERROR: keras version MUST be 1.1.0"
import numpy as np
from keras import backend as K
from theano import tensor
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Permute, Reshape
from keras.layers import Convolution2D, AveragePooling2D, SpatialDropout2D
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import ZeroPadding2D, UpSampling2D, Input, merge
from keras.models import Model
from keras.utils.np_utils import convert_kernel
#--------------------------------------------------------------------------------
# Customized Softmax Function along the filter dimension
#--------------------------------------------------------------------------------
def softmax4(x):
    """Custom softmax activation function for a 4D input tensor
    softmax along axis = 1
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim == 3:
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    elif ndim == 4:
        e = K.exp(x - K.max(x, axis=1, keepdims=True))
        s = K.sum(e, axis=1, keepdims=True)
        return e / s
    else:
        raise Exception('Cannot apply softmax to a tensor that is not 2D or 3D. ' +
                        'Here, ndim=' + str(ndim))

#--------------------------------------------------------------------------------
# PPT Model Definition - single resolution
#--------------------------------------------------------------------------------
def create_single_res_model() :
    '''Create a PPT text detector model with single resolution support
    '''
    model = Sequential()
    # block 1
    model.add( ZeroPadding2D( padding = ( 3, 3 ), input_shape = ( 3, None, None ) ) )
    model.add( Convolution2D( 16, 7, 7, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 2, 2 ) ) )
    model.add( Convolution2D( 16, 5, 5, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 2
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 2, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 2, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 2, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 2, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 3
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 4, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 4, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 4, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 16 * 4, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 4
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16 * 8, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 0, 1 ) ) )
    model.add( Convolution2D( 16 * 16, 1, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 3, 1, 1, border_mode='valid' ) )
    model.add( Activation( 'relu' ) )
    # block 5
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 2, 2 ) ) )
    model.add( Convolution2D( 12, 5, 5, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 3, 3 ) ) )
    model.add( Convolution2D( 8, 7, 7, border_mode='valid' ) )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 5, 5 ) ) )
    model.add( Convolution2D( 3, 11, 11, border_mode='valid' ) )
    model.add( Activation( softmax4 ) )
    return model

#--------------------------------------------------------------------------------
# PPT Model Definition - multiple resolution
#--------------------------------------------------------------------------------
def th2tf( model):  
    import tensorflow as tf  
    ops = []  
    for layer in model.layers:  
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:  
            original_w = K.get_value(layer.W)  
            converted_w = convert_kernel(original_w)  
            #converted_w = original_w.transpose(0,1,3,2) 
            ops.append(tf.assign(layer.W, converted_w).op)  
    K.get_session().run(ops)  
    return model  
  
def tf2th(model):  
    for layer in model.layers:  
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:  
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w) 
            #print original_w.shape
            converted_w = original_w.transpose(0,1,3,2)  
            K.set_value(layer.W, converted_w)  
    return model  
  
def conv_layer_converted(model,th_weights, tf_weights, m = 1):  
    """ 
    :param tf_weights: 
    :param th_weights: 
    :param m: 0-tf2th, 1-th2tf 
    :return: 
    """  
    if m == 0: # tf2th   
        model = tf2th(model)  
        model.save_weights(th_weights)  
    elif m == 1: # th2tf  
        model = th2tf(model)  
        model.save_weights(tf_weights)  
    else:  
        print("0-tf2th, 1-th2tf")  
        return 




def create_multi_res_model() :
    '''Create a PPT text detector model using multi-resolution responses
    '''
    base_model = create_single_res_model()
    # create a multi-resolution model
    inputs = Input( shape = ( 3, None, None ) )
    a2 = AveragePooling2D((2,2))( inputs )
    a3 = AveragePooling2D((3,3))( inputs )
    a4 = AveragePooling2D((4,4))( inputs )
    # decode at each resolution
    p1 = base_model( inputs )
    p2 = base_model( a2 )
    p3 = base_model( a3 )
    p4 = base_model( a4 )
    # dropout
    d1 = SpatialDropout2D(0.25)(p1)
    d2 = SpatialDropout2D(0.25)(p2)
    d3 = SpatialDropout2D(0.25)(p3)
    d4 = SpatialDropout2D(0.25)(p4)
    # map to original resolution
    o2 = UpSampling2D((2,2))(d2)
    o3 = UpSampling2D((3,3))(d3)
    o4 = UpSampling2D((4,4))(d4)
    # merge all response
    f = merge( [ d1, o2, o3, o4 ], mode = 'concat', concat_axis = 1 )
    f_pad = ZeroPadding2D((5,5))(f)
    bottle = Convolution2D(8, 11, 11, activation='relu', name = 'bottle' )( f_pad )
    output = Convolution2D(3, 1, 1, activation=softmax4 )( bottle )
    model = Model( input = inputs, output = output )
    return model


def cross_entropy( true_dist, coding_dist ) :
    return -tensor.sum(true_dist * tensor.log(coding_dist), axis=1 )
def create_model( ) :
    # Define CNN model
    model = Sequential()
    # block 1
    model.add( ZeroPadding2D( padding = ( 3, 3 ), input_shape = ( 3, 256, 336 ) ) )
    model.add( Convolution2D( 16, 7, 7, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 32, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 2, 2 ) ) )
    model.add( Convolution2D( 48, 5, 5, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 64, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 2
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 80, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 96, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 112, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 128, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 3
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 144, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 160, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 176, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 192, 1, 1, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    # block 4
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 224, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( ZeroPadding2D( padding = ( 0, 1 ) ) )
    model.add( Convolution2D( 256, 1, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( Convolution2D( 3, 1, 1, border_mode='valid' ) )
    model.add( Activation( 'relu' ) )
    # block 5
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 1, 1 ) ) )
    model.add( Convolution2D( 16, 3, 3, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 3, 3 ) ) )
    model.add( Convolution2D( 8, 7, 7, border_mode='valid' ) )
    model.add( Activation('relu') )
    model.add( UpSampling2D( ( 2, 2 ) ) )
    model.add( ZeroPadding2D( padding = ( 5, 5 ) ) )
    model.add( Convolution2D( 3, 11, 11, border_mode='valid' ) )
    model.add( Activation( softmax4 ) )
    model.compile('adam', cross_entropy )
    return model



if __name__ == '__main__':
    model = create_multi_res_model()
    model.load_weights('model/multi_res.h5')
    conv_layer_converted(model,'model/multi_res_th.h5','model/multi_res_tf.h5')
    #weight_format()