# -*- coding: utf-8 -*-
'''
Self-Organized Text Detection with Mimimal Postprocessing with Border Learning

This script wraps baseline single/multi-resolution PPT text detectors to command-
line tool.

Created on Thu Jul 20 19:34:30 2017

@author: yue_wu
'''

import os
import argparse
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.4f')
from skimage.morphology import label as bwlabel
import warnings
warnings.filterwarnings("ignore")
file_path = os.path.realpath( __file__ )
repo_root = os.path.join( os.path.dirname( file_path ), os.path.pardir )
_default_multi_res_weight = os.path.join( repo_root, 'model', 'multi_res.h5' )
from utils import imread, imwrite, prepare_input, decode_image
from models import create_single_res_model, create_multi_res_model

def verbose_print( message, prog_verbose, thresh_verbose ) :
    if ( prog_verbose <= thresh_verbose ) :
        return
    else :
        if ( isinstance( message, list ) ) :
            message = " ".join( [ str( elem ) for elem in message ] )
        print message

def prepare_model( res_type, weight_file ) :
    try :
        text_detector = create_single_res_model() if ( res_type == 'single' ) \
                            else create_multi_res_model()
        text_detector.load_weights( weight_file )
    except Exception, e :
        raise IOError, "ERROR: fail to create %s model or load weight from %s, %s" % \
            ( res_type, weight_file, str(e) )
    return text_detector

def prepare_output_dir( output_dir ) :
    if ( os.path.isdir( output_dir ) ) :
        return
    else :
        try:
            os.makedirs( output_dir )
        except OSError as err:
            if err.errno!=17:
                raise IOError, "ERROR: cannot locate/create output_dir %s" % output_dir
    return

def relax_wrt_border( raw_bbox, height, width, border_perc = .16 ) :
    # load bbox
    top, bot, left, right = raw_bbox
    # compute box width and height
    box_w, box_h = right - left + 1, bot - top + 1
    # compute border width
    d = np.ceil( min( box_w, box_h ) * ( 1./ ( 1 - border_perc * 2 ) - 1 ) * 0.5 )
    # relax according to border info
    left, right  = max( 0, left - d ), min( right + d, width )
    top,  bot    = max( 0, top - d ), min( bot + d, height )
    return [ top, bot, left, right ]

def from_res_map_to_bbox( res_map, th_size = 8, th_prob = 0.25, border_perc = .16 ) :
    height, width = res_map.shape[:2]
    labels = res_map.argmax( axis = -1 )
    text = labels == 2
    bord = labels == 1
    
    #including bord as relax for bbox (can completely contain the text that we need)
    text = text | bord
    
    bwtext, nb_regs = bwlabel( text, return_num = True )
    #bwrelax , nb_regs = bwlabel( relax, return_num = True )
    lut = { 'bounding_box' : [], 'proba' : [] }
    for reg_id in range( 1, nb_regs + 1 ) :
        row_idx, col_idx = np.nonzero( bwtext == reg_id )
        # get four corners
        left, right = col_idx.min(), col_idx.max() + 1
        top, bot = row_idx.min(), row_idx.max() + 1
        # relax w.r.t. border
        bbox = relax_wrt_border( [ top, bot, left, right ], height, width, border_perc )
        by0, by1, bx0, bx1 = bbox
        bh, bw = by1 - by0 + 1, bx1 - bx0 + 1
        # estimate text proba
        # because of relax,the rectangle have some pixels' value are zero
        # so the pro may be very small although it was text region
        proba = np.median( res_map[top:bot, left:right, 2 ] )
        if ( proba >= th_prob ) and ( min( bh, bw ) >= th_size ):
            lut['bounding_box'].append( [ left, top, right, bot ] )
            lut['proba'].append( float( proba ) )
    return lut

def safe_decoder( model, image_file, res_type ) :
    try :
        if ( not os.path.isfile( image_file ) ) :
            raise IOError('fail to locate image_file')
        # load image array from file
        image_array = imread( image_file )
        if ( image_array is None ) :
            raise IOError('fail to decode image_file')
        # prepare image array as theano tensor
        image_shape = image_array.shape[:2]
        th_tensor = prepare_input( image_array, res_type )
        if ( th_tensor.mean() < 0 ) :
            th_tensor = -th_tensor
        # decode input using the pretrained PPT text detector
        res_map = decode_image( model, th_tensor, image_shape )
    except Exception, e :
        print "WARNING: something wrong during decoding", image_file, e
        res_map = None
    return res_map


if __name__ == '__main__' :
    '''PPT Text Detection with Border Support
    Usage:
        ppt_text_detector.py -h
    '''
    parser = argparse.ArgumentParser( description = 'PPT Text Detection with Border Support' )
    parser.add_argument( '-i', action = 'append', dest = 'input_files', default = [], help = 'input test image files' )
    parser.add_argument( '-o', action = 'store', dest = 'output_dir', default = './', help = 'output detection dir (./)' )
    parser.add_argument( '-w', action = 'store', dest = 'weight_file', default = None, help = 'keras model weight file (model/multi_res.h5)' )
    parser.add_argument( '-v', action = 'store', dest = 'verbose', type = int, default = 0, help = 'verbosity level (0), higher means more print outs')
    parser.add_argument( '-th_size', action = 'store', dest = 'th_size', type = int, default = 8, help = 'minimal text line dimension (8) in output' )
    parser.add_argument( '-th_prob', action = 'store', dest = 'th_prob', type = float, default = .25, help = 'minimal text line probability (.25) in output' )
    parser.add_argument( '--no_map', action = 'store_true', dest = 'no_map', default = False, help = 'not produce text detection response maps' )
    parser.add_argument( '--no_box', action = 'store_true', dest = 'no_box', default = False, help = 'not save individual text bounding boxes to json')
    parser.add_argument( '--version', action = 'version', version = '%(prog)s 1.0' )
    #----------------------------------------------------------------------------
    # parse program arguments
    #----------------------------------------------------------------------------
    args = parser.parse_args()
    input_files = args.input_files
    output_dir = args.output_dir
    weight_file = args.weight_file
    th_size = args.th_size
    th_prob = args.th_prob
    no_map = args.no_map
    no_box = args.no_box
    verbose = args.verbose
    if ( weight_file is None ) :
        weight_file = _default_multi_res_weight
    # print program I/O
    all_file_str = ','.join( [ f for f in input_files ] )
    if ( len( all_file_str ) > 100 ) :
        all_file_str = '\n'.join( [ '\t\t'+f for f in input_files ] ) + '\n'
    io_message = '\n'.join( [ '-' * 100,
                              '-' + 'Program I/O',
                              '-' * 100,
                              'input_files = %s' % all_file_str,
                              'output_dir = %s' % output_dir,
                              'weight_file = %s' % weight_file,
                              'thresh_size = %d' % th_size,
                              'thresh_prob = %.2f' % th_prob,
                              'no_map = %d' % no_map,
                              'no_box = %d' % no_box,
                              'verbose = %d' % verbose ] ) + '\n'
    verbose_print( io_message, verbose, 0 )
    #----------------------------------------------------------------------------
    # prepare output dir and text detector
    #----------------------------------------------------------------------------
    prepare_output_dir( output_dir )
    res_type = 'multiple'
    text_detector = prepare_model( res_type, weight_file )
    #----------------------------------------------------------------------------
    # text detection for all images
    #----------------------------------------------------------------------------
    all_luts = []
    for image_file in input_files :
        res_map = safe_decoder( text_detector, image_file, res_type )
        if ( res_map is not None ) :
            file_bname = os.path.basename( image_file ).rsplit('.')[0]
            lut = { 'input_file' : image_file }
            blut = from_res_map_to_bbox( res_map, th_size = th_size, th_prob = th_prob, border_perc = .16 )
            lut.update( blut )
            # write response map if necessary
            if ( not no_map ) :
                output_res_file = os.path.join( output_dir, file_bname + '.png' )
                lut[ 'response_map' ] = output_res_file
                imwrite( np.round( res_map * 255 ).astype( 'uint8' ), output_res_file )
                verbose_print( ['INFO: dump response map to', output_res_file ], verbose, 1 )
            # write output detection json file if necessary
            if ( not no_box ) :
                output_box_json = os.path.join( output_dir, file_bname + '.json' )
                json.dump( lut, open( output_box_json, 'w' ), indent = 4 )
                verbose_print( ['INFO: dump text bbox to', output_box_json ], verbose, 1 )
            all_luts.append( lut )
    if ( no_box ) :
        output_mega_json = os.path.join( output_dir, 'decoded_bbox.json' )
        json.dump( all_luts, open( output_mega_json, 'w' ), indent = 4 )
        verbose_print( ['INFO: dump mega text bbox to', output_mega_json ], verbose, 1 )
        
