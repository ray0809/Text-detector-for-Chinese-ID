# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:42:14 2017

@author: yue_wu

This script defines axillary functions for ISI-PPT dataset
"""
import numpy as np 
import os
import shutil




# load image I/O lib
use_cv2 = False
use_PIL = False
try :
    import cv2
    use_cv2 = True
except Exception, e :
    import PIL
    use_PIL = True

assert ( use_cv2 or use_PIL  ), "ERROR: cannot load any known common Py2 Image Libs to read and write images\nSupported Libs: cv2, PIL"
    
def imread( image_file_path ) :
    '''Unified image read function
    INPUT:
        image_file_path = string, path to image file
    OUTPUT:
        image_array = np.ndarray, shape of ( height, width, 3 ), dtype of uint8
    '''
    assert os.path.isfile( image_file_path ), "ERROR: cannot locate image file %s" % image_file_path 
    if ( use_cv2 ) :
        image_array = cv2.imread( image_file_path, 1 )
    elif ( use_PIL ) :
        image_array = np.array( PIL.Image.open( image_file_path ) )
    if ( image_array.ndim != 3 ) :
        image_array = np.dstack( [ image_array for k in range(3) ] ) # convert a grayscale image to RGB
    return image_array

def imwrite( image_array, output_file_path ) :
    '''Unified image write function
    INPUT:
        image_array = np.ndarray, shape of ( height, width, 3 ), dtype of uint8
        output_file_path = string, path to image file
    OUTPUT:
        status = bool, True if write successfully, False otherwise
    '''
    if ( use_cv2 ) :
        status = cv2.imwrite( output_file_path, image_array )
    elif ( use_PIL ):
        pil_image = PIL.Image.fromarray( image_array )
        pil_image.save( output_file_path )
        status = os.path.isfile( output_file_path )
    return status

def prepare_input( image_array, res_type = 'multiple' ) :
    '''Prepare input image array to theano tensor
    '''
    if ( res_type == 'single' ) :
        multiple_of_X = 8
    elif ( res_type == 'multiple' ) :
        multiple_of_X = 96
    else :
        raise NotImplementedError, "ERROR: res_type = %s is NOT supported" % res_type
    # determine padding patterns
    h, w = image_array.shape[:2]
    pad_h = ( h // multiple_of_X * multiple_of_X - h ) % multiple_of_X
    pad_w = ( w // multiple_of_X * multiple_of_X - w ) % multiple_of_X
    # pad image to make sure the new dimension is a multiple of ${multiple_of_X}
    image_pad = np.pad( image_array, ( [ 0, pad_h ], [ 0, pad_w ], [ 0, 0 ] ),
                        mode = 'symmetric')
    th_tensor = image_pad.astype( np.float32 )
    th_tensor = th_tensor.transpose(2,0,1)
    # convert image uint8 array to theano float32 tensor
    # th_tensor = np.rollaxis( image_pad.astype( np.float32 ), 2, 0 )
    th_tensor = np.expand_dims( th_tensor, 0 ) / 255. - 0.5
    return th_tensor

def decode_image( model, th_tensor, output_shape ) :
    '''Decode a PPT image using an existing model
    '''
    h, w = output_shape
    res_map = model.predict( th_tensor )
    #res_map = np.rollaxis( res_map[0], 0, 3 )
    res_map = res_map[:h,:w]
    return res_map










"""
Created on Mon Aug 14 15:00:10 2017

@author: ray
"""


def create_list():
    imgs = os.listdir('data/')
    with open('data_list.list','a') as f:
        for img in imgs:
            if img.endswith('jpg'):
                label = img.replace('jpg','json')
                if label in imgs:
                    shutil.move('data/'+label,'label/'+label)
                f.write('data/' + img + ' ' + 'label/' + label + '\n')

    


if __name__ == '__main__':
    create_list()

        
    
