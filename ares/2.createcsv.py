#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
vision:python3
"""
import os
import math  
import numpy as np  
import tensorflow as tf
from sys import argv

patchNumbers = 12
np.random.seed()

def get_files(PG_file_dir, CG_file_dir, test_ratio, val_ratio):
    image_list0 = []
    image_list1 = []
    for file in os.listdir(CG_file_dir):  
        image_list0.append(CG_file_dir +'/'+ file)  
    for file in os.listdir(PG_file_dir):  
        image_list1.append(PG_file_dir +'/'+ file)
    image_list0.sort()
    image_list1.sort()

    #split to train, val, test
    #n_sample = min(len(image_list0),len(image_list1))/patchNumbers
    n_sample = 360
    n_test = int(math.ceil(n_sample))
    #n_val = int(math.ceil(n_sample*val_ratio))
    #n_train = n_sample - n_val - n_test
    n_train = 0
    n_val = 0 
    #print n_sample*2, n_train*2, n_val*2, n_test*2

    #shuffle and split
    idx = [i for i in range(n_sample)]
    np.random.shuffle(idx)
    '''
    train_images = []
    train_labels = []
    for i in idx[0:n_train]:
        train_images += image_list0[i*patchNumbers : i*patchNumbers+patchNumbers]
        train_labels += [0]*patchNumbers
        train_images += image_list1[i*patchNumbers : i*patchNumbers+patchNumbers]
        train_labels += [1]*patchNumbers
    val_images = []
    val_labels = []
    for i in idx[n_train:n_train+n_val]:
        val_images += image_list0[i*patchNumbers : i*patchNumbers+patchNumbers]
        val_labels += [0]*patchNumbers
        val_images += image_list1[i*patchNumbers : i*patchNumbers+patchNumbers]
        val_labels += [1]*patchNumbers
    '''
    test_images = []
    test_labels = []
    for i in idx[0:]:
        test_images += image_list0[i*patchNumbers : i*patchNumbers+patchNumbers]
        test_labels += [1 0]*patchNumbers
        test_images += image_list1[i*patchNumbers : i*patchNumbers+patchNumbers]
        test_labels += [0 1]*patchNumbers

    return  test_images, test_labels


if __name__ == '__main__':
    #main code
    if (len(argv)<3):
        #print "usage: python createcsv.py PGFilePath CGFilePath\n"
        exit()

    test_images, test_labels = get_files(
        PG_file_dir = argv[1],
        CG_file_dir = argv[2],
        test_ratio=1, val_ratio=0)
    #print train_images[0:10]
    #print train_labels[0:10]
    '''
    with open("train_list.csv", "w") as fid:
        for i in range(len(train_labels)):
            strText = train_images[i] + "," + str(train_labels[i]) + "\n"
            fid.write(strText)
        fid.close()

    with open("val_list.csv", "w") as fid:
        for i in range(len(val_labels)):
            strText = val_images[i] + "," + str(val_labels[i]) + "\n"
            fid.write(strText)
        fid.close()
    '''
    with open("test_list10.csv", "w") as fid:
        for i in range(len(test_labels)):
            strText = test_images[i] + "," + str(test_labels[i]) + "\n"
            fid.write(strText)
        fid.close()
    print('Done')
