
import cv2
import os
import numpy as np
import os, os.path
import random
from sys import argv
import numpy as np

rectWidth = 224  # 256*256
patchNumbers = 20 

def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
 
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
 
                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
 
                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img


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
    if (imgw<360 or imgh<640):
        print(file_name, "file too small")
        return
    img = bilinear_interpolation(img,(640,360))
    blockcnt = 0
    i=j=0
    for i in range(12):
        px = random.randint(0, 360-rectWidth)
        py = random.randint(0, 640-rectWidth)

        #save rect image to file
        cropImg = img[px:px+rectWidth , py:py+rectWidth]
        tName = "%s/%s#%s.bmp" %(output_path, file_name.split('.')[0], str(blockcnt).zfill(4))
        #cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(tName, cropImg)
        #cv2.imwrite(outPathName, img, [int( cv2.IMWRITE_JPEG_QUALITY), 70])
        blockcnt += 1
        
def cut_CG(input_path, output_path, file_name):
    img = cv2.imread(input_path + '/' + file_name)
    imgw = img.shape[1]
    imgh = img.shape[0]
    if (imgw<224 or imgh<224):
        print(file_name, "file too small")
        return
    #img = bilinear_interpolation(img)
    blockcnt = 0
    i=j=0
    for i in range(12):
        px = random.randint(0, imgh-rectWidth)
        py = random.randint(0, imgw-rectWidth)

        #save rect image to file
        cropImg = img[px:px +rectWidth , py:py+rectWidth]
        tName = "%s/%s#%s.bmp" %(output_path, file_name.split('.')[0], str(blockcnt).zfill(4))
        #cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(tName, cropImg)
        #cv2.imwrite(outPathName, img, [int( cv2.IMWRITE_JPEG_QUALITY), 70])
        blockcnt += 1
                

def cut_image_patches(source_real, source_CG, patch_dir='..'): 

    #make dirs
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)
        os.mkdir(patch_dir + '/train/')
        os.mkdir(patch_dir + '/test/')
        os.mkdir(patch_dir + '/valid/')
        os.mkdir(patch_dir + '/train/CGG/')
        os.mkdir(patch_dir + '/test/Real/')
        os.mkdir(patch_dir + '/train/Real/')
        os.mkdir(patch_dir + '/test/CGG/')
        os.mkdir(patch_dir + '/valid/Real/')
        os.mkdir(patch_dir + '/valid/CGG/')
    

    image_real, number_real = load_images_from_dir(source_real, shuffle = True)
    image_CG, number_cg = load_images_from_dir(source_CG, shuffle = True)
    if (number_cg != number_real):
        print('different number of images')
        exit()
    k = 0
    for i in range(number_real):
        if k < 7:
            cut_one_image_random(source_real, patch_dir + '/train/Real/', image_real[i])
            cut_CG(source_CG, patch_dir + '/train/CGG/', image_CG[i])
            k+=1
        elif k < 9 :
            cut_one_image_random(source_real, patch_dir + '/test/Real/', image_real[i])
            cut_CG(source_CG, patch_dir + '/test/CGG/', image_CG[i])
            k+= 1 
        elif  k == 9 :
            cut_one_image_random(source_real, patch_dir + '/valid/Real/', image_real[i])
            cut_CG(source_CG, patch_dir + '/valid/CGG/', image_CG[i])
            k = 0

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
    

