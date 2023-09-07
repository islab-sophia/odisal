"""
extract 2D images from Omni-Directional Images
"""
import argparse
import configparser
import csv
import datetime
import os
import pickle
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pylab import *
from skimage.transform import resize
from tqdm import tqdm

import odimapping

Image.MAX_IMAGE_PIXELS = 200000000

def parse_args():
    parser = argparse.ArgumentParser(description='extract 2D images from Omni-Directional Images')
    parser.add_argument('--extract', default='E26', type=str,
                        help='method of extraction')
    parser.add_argument('--maps-dir', default='maps',
                        type=str, help='name of maps-dir')

    parser.add_argument('--csv-path', default='/home/lifeichen/DataSets/salient360_2018/salient360_2018.csv',
                        type=str, help='path to csv file of odi dataset')
    parser.add_argument('--output-root-dir', default='/home/lifeichen/DataSets/',
                        type=str, help='path to root dir of output extracted dataset')


    parser.add_argument('--input_w', default=1600, type=int,
                        help='resizing the width of the input image')
    parser.add_argument('--input_h', default=800, type=int,
                        help='resizing the height of the input image')
    parser.add_argument('--extract_w', default=500, type=int,
                        help='resizing the width of the input image')
    parser.add_argument('--extract_h', default=500, type=int,
                        help='resizing the height of the input image')

    parser.add_argument('--view_angle_list', default='80,100,120', type=str,
                        help='setting the view angle of the input image')
                        
    parser.add_argument('--salient360_2018', action='store_true',
                        help='use salient360_2018 dataset')
    return parser.parse_args()

def main(args):
    time_stamp = "{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())
    parent_dataset_name = args.csv_path.split('/')[-1].split('.')[0]
    mapping_name = "{}_{}_{}_{}_{}".format(args.extract, args.input_w, args.input_h, args.extract_w, args.extract_h)
    view_angle_list = [int(i) for i in args.view_angle_list.split(',')]

    for i in range(len(view_angle_list)):
        mapping_name = mapping_name + '_' + str(view_angle_list[i])
    output_dataset_name = "{}_{}".format(parent_dataset_name, mapping_name)

    save_dir = os.path.join(args.output_root_dir, output_dataset_name)
    print("input dataset name : {}".format(parent_dataset_name))
    print("mapping name : {}".format(mapping_name))
    print("output dataset name : {}".format(output_dataset_name))
    print("save dir : {}".format(save_dir))

    filepath_df = pd.read_csv(args.csv_path)

    output_imgs_dir = os.path.join(save_dir, "imgs")
    output_he_salmap_dir = os.path.join(save_dir, "he_salmap")
    output_csv_path = os.path.join(save_dir, "{}.csv".format(output_dataset_name))
    print("output csv path : {}".format(output_csv_path))
    config_path = os.path.join(save_dir, "{}.cfg".format(output_dataset_name))
    constant_bias_path = os.path.join(save_dir, "{}_constant_bias.npy".format(output_dataset_name))
    extracted_constant_bias_path = os.path.join(save_dir, "{}_extracted_constant_bias.npy".format(output_dataset_name))
    args_path = os.path.join(save_dir, "{}_args.txt".format(output_dataset_name))
    if not os.path.isdir(output_imgs_dir):
        os.makedirs(output_imgs_dir)
    if not os.path.isdir(output_he_salmap_dir):
        os.makedirs(output_he_salmap_dir)

    with open(args_path, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    
    output_filepath_df = pd.DataFrame(index=[], columns=[
        'parent_number',
        'view_angle',
        'extract_number',
        'img_path',
        'he_salmap_im_path',
        'train',
        'val',
        'test',
        'parent_img_path',
        'parent_he_salmap_bin_path',
        'output_dataset_name'
    ])

    mapping = odimapping.equirectangular(args.extract, args.maps_dir, view_angle_p=view_angle_list[1], view_angle_t=view_angle_list[1], extract_h=args.extract_h, extract_w=args.extract_w, odi_h=args.input_h, odi_w=args.input_w)
    RGB = np.empty((3,len(filepath_df)*mapping.num_extracted*len(view_angle_list)))
    k = 0
    extracted_constant_bias_list = np.zeros((mapping.num_vertical, mapping.extract_size[0], mapping.extract_size[1], 1))

    for t in range(len(view_angle_list)):
        print("when view angle = ({}, {})".format(view_angle_list[t], view_angle_list[t]))
        mapping = odimapping.equirectangular(args.extract, args.maps_dir, view_angle_p=view_angle_list[t], view_angle_t=view_angle_list[t], extract_h=args.extract_h, extract_w=args.extract_w, odi_h=args.input_h, odi_w=args.input_w)
        
        constant_bias = np.zeros((int(args.input_h), int(args.input_w)))
        for i in tqdm(range(len(filepath_df))):
            # print("i {}/{}".format(i, len(filepath_df)))
            parent_number = filepath_df["number"][i]
            parent_img_path = filepath_df["img_path"][i]
            parent_he_salmap_bin_path = filepath_df["he_salmap_bin_path"][i]
            train_flag = filepath_df["train"][i]
            val_flag = filepath_df["val"][i]
            test_flag = filepath_df["test"][i]

            parent_im = Image.open(parent_img_path)
            original_h = parent_im.size[1]
            original_w = parent_im.size[0]
            odi_im_ndarray = np.expand_dims(np.array(parent_im.resize((args.input_w, args.input_h)), dtype=np.float32), axis=0)
            extracted_im_ndarray = mapping.extract(odi_im_ndarray)
            
            if args.salient360_2018:
                #salient360_2018
                original_w = 2048
                original_h = 1024
            num_bin_loop = original_h*original_w

            if os.path.isfile(parent_he_salmap_bin_path+'.npy'):
                odi_salmap_ndarray = np.load(parent_he_salmap_bin_path+'.npy')
            else:
                odi_salmap_ndarray = np.empty(original_w*original_h).astype(np.float)
                with open(parent_he_salmap_bin_path, "rb") as f:
                    for j in range(num_bin_loop):
                        b = f.read(4)
                        c = struct.unpack("f", b)
                        odi_salmap_ndarray[j] = c[0]
                odi_salmap_ndarray = odi_salmap_ndarray.reshape((original_h, original_w))
                np.save(parent_he_salmap_bin_path+".npy" ,odi_salmap_ndarray)

            odi_salmap_ndarray = resize(odi_salmap_ndarray, (args.input_h, args.input_w), order=0, mode="constant")
            odi_salmap_ndarray = odi_salmap_ndarray/odi_salmap_ndarray.sum()

            if train_flag:
                constant_bias += odi_salmap_ndarray

            odi_salmap_ndarray = (odi_salmap_ndarray-odi_salmap_ndarray.min())/(odi_salmap_ndarray.max()-odi_salmap_ndarray.min())
            odi_salmap_ndarray = odi_salmap_ndarray*255
            Image.fromarray(np.uint8(odi_salmap_ndarray)).save(os.path.join(output_he_salmap_dir, str(parent_number)+'.jpg'))

            extracted_salmap_ndarray = mapping.extract(odi_salmap_ndarray[np.newaxis, :, :, np.newaxis])

            for j in range(mapping.num_extracted):
                extracted_im = Image.fromarray(np.uint8(extracted_im_ndarray[j]))
                extracted_salmap = Image.fromarray(np.uint8(extracted_salmap_ndarray[j, :, :, 0]))
                extract_number = j
                img_path = os.path.join(output_imgs_dir, str(parent_number)+'_'+str(view_angle_list[t])+'_'+str(extract_number)+'.jpg')
                he_salmap_im_path = os.path.join(output_he_salmap_dir, str(parent_number)+'_'+str(view_angle_list[t])+'_'+str(extract_number)+'.jpg')
                extracted_im.save(img_path)
                extracted_salmap.save(he_salmap_im_path)

                extracted_im_t = np.array(extracted_im).transpose(2,0,1) #TODO
                RGB[0][k] = extracted_im_t[0].mean()
                RGB[1][k] = extracted_im_t[1].mean()
                RGB[2][k] = extracted_im_t[2].mean()
                k += 1

                tmp = pd.Series([
                    parent_number,
                    view_angle_list[t],
                    extract_number,
                    img_path,
                    he_salmap_im_path,
                    train_flag,
                    val_flag,
                    test_flag,
                    parent_img_path,
                    parent_he_salmap_bin_path,
                    output_dataset_name], 
                    index=['parent_number',
                        'view_angle',
                        'extract_number',
                        'img_path',
                        'he_salmap_im_path',
                        'train',
                        'val',
                        'test',
                        'parent_img_path',
                        'parent_he_salmap_bin_path',
                        'output_dataset_name'])
                output_filepath_df = output_filepath_df.append(tmp, ignore_index=True)
            output_filepath_df.to_csv(output_csv_path, index=False)

        constant_bias /= filepath_df["train"].sum()
        extracted_constant_bias = mapping.extract(constant_bias[np.newaxis, :, :, np.newaxis])
        tmp_extracted_constant_bias_list = np.zeros((mapping.num_vertical, mapping.extract_size[0], mapping.extract_size[1], 1))
        for m in range(mapping.num_vertical):
            if m==0:
                tmp_extracted_constant_bias_list[m] = extracted_constant_bias[0]
            elif m==(mapping.num_vertical-1):
                tmp_extracted_constant_bias_list[m] = extracted_constant_bias[1]
            else:
                for n in range(2+(m-1)*mapping.num_horizontal, 2+m*mapping.num_horizontal):
                    tmp_extracted_constant_bias_list[m] += extracted_constant_bias[n]
                tmp_extracted_constant_bias_list[m] /= mapping.num_horizontal
                   
            extracted_constant_bias_list[m] +=  tmp_extracted_constant_bias_list[m]

    mean_r = RGB[0].mean()
    mean_g = RGB[1].mean()
    mean_b = RGB[2].mean()
    
    for i in range(mapping.num_vertical):
        extracted_constant_bias_list[i] /= len(view_angle_list)

    np.save(constant_bias_path, constant_bias)
    np.save(extracted_constant_bias_path, extracted_constant_bias_list)

    config = configparser.ConfigParser()
    config['dataset'] = {'dataset_name' : output_dataset_name,
                        'time_stamp' : time_stamp,
                        'mapping_name' : mapping_name,
                        'parent_dataset_name' : parent_dataset_name,
                        'csv_path': output_csv_path,
                        'mean_r': mean_r,
                        'mean_g': mean_g,
                        'mean_b': mean_b,
                        'input_h' : args.input_h,
                        'inptu_w' : args.input_w,
                        'extract_h' : args.extract_h,
                        'extract_w' : args.extract_w,
                        'num_vertical' : str(mapping.num_vertical),
                        'num_horizontal' : str(mapping.num_horizontal),
                        'num_extracted' : str(mapping.num_extracted),
                        'view_angle_p' : str(mapping.view_angle[0]),
                        'view_angle_t' : str(mapping.view_angle[1])
                        }
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    print("config file : {}".format(config_path))

if __name__ == '__main__':
    args = parse_args()
    main(args)
