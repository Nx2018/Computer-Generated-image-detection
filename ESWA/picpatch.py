
import cv2
import os
import numpy as np
import os, os.path
import random
from sys import argv
import numpy as np

rectWidth = 227  #image size
patchNumbers = 20 #

def load_images_from_dir(dir_name, shuffle = True) :
    # file extension accepted as image data
    valid_image_extension = [".jpg", ".jpeg", ".png", ".tif", ".JPG"]

    file_list = []
    nb_image = 0
    for filename in os.listdir(dir_name):
        # check if the file is an image
        extension = os.path.splitext(filename)[1]
        if extension.lower() in valid_image_extension:
            file_list.append(filename)
            nb_image += 1

    print('    ', nb_image, 'images loaded')

    if shuffle:
        random.shuffle(file_list)
    return file_list, nb_image


def cut_one_image_random(input_path, output_path, file_name):
    img = cv2.imread(input_path + '/' + file_name)
    imgw = img.shape[1]
    imgh = img.shape[0]
    if (imgw<rectWidth or imgh<rectWidth):
        print file_name, "file too small"
        return

    blockcnt = 0
    while (blockcnt<patchNumbers):
        px = random.randint(0, imgw-rectWidth)
        py = random.randint(0, imgh-rectWidth)
        #print px, py

        #save rect image to file
        cropImg = img[py:py+rectWidth, px:px+rectWidth]
        tName = "%s/%s#%s.bmp" %(output_path, file_name.split('.')[0], str(blockcnt).zfill(4))
        #cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(tName, cropImg)
        #cv2.imwrite(outPathName, img, [int( cv2.IMWRITE_JPEG_QUALITY), 70])
        blockcnt += 1
    
        
        

def cut_image_patches(source_real, source_CG, patch_dir='..'): 

    #make dirs
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)
        os.mkdir(patch_dir + '/CGG/')
        os.mkdir(patch_dir + '/Real/')
        
    

    image_real, number_real = load_images_from_dir(source_real, shuffle = True)
    image_CG, number_cg = load_images_from_dir(source_CG, shuffle = True)
    if (number_cg != number_real):
        print('different number of images')
        exit()
    k = 0
    for i in range(number_real): 
      
        cut_one_image_random(source_real, patch_dir + '/Real/', image_real[i])
        cut_one_image_random(source_CG, patch_dir + '/CGG/', image_CG[i])
            
            
    print(str(number_real) + 'images save to ' + patch_dir)
    print("Done.")


if __name__ == '__main__': 
    #main code
    if (len(argv)<4):
        print("usage: python cutpatch.py PGFilePath CGFilePath outpatchdir\n")
        exit()

    # Change to the source of real images
    source_real_directory = argv[1] #"/home/forensics/CGvsPG/dataset/jvci2013_PG4850/"
    # Change to the source of CG images
    source_CG_directory = argv[2] #"/home/forensics/CGvsPG/dataset/jvci2013_CG4850/"
    # The directory where the database will be saved
    patch_dir_test = argv[3] #'/home/forensics/CGvsPG/imagepatches/'

    #random.seed(42)
    random.seed()

    cut_image_patches(source_real = source_real_directory, 
              source_CG = source_CG_directory,
              patch_dir = patch_dir_test)
    
