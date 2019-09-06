import cv2
import os
import numpy as np
import os, os.path
import random

rectWidth = 650
setpSize = 650

def load_images_from_dir(dir_name, shuffle = True) :
    # file extension accepted as image data
    valid_image_extension = [".jpg", ".jpeg", ".png", ".tif", ".JPG"]

    file_list = []
    nb_image = 0 
    for filename in os.listdir(dir_name):
        # check if the file is an image
        extension = os.path.splitext(filename)[1]
        if extension.lower() in valid_image_extension:
            file_list.append(os.path.splitext(filename)[0])
            nb_image += 1

    print('    ',nb_image,'images loaded')

    if shuffle: 
        #random.seed(42)
        random.seed()
        random.shuffle(file_list)
    return file_list


def make_dirs(target_dir):

    train_dir = target_dir + '/train/'
    test_dir = target_dir + '/test/'
    validation_dir = target_dir + '/validation/'

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        os.mkdir(train_dir)
        os.mkdir(train_dir + 'CGG/')
        os.mkdir(train_dir + 'Real/')
        os.mkdir(test_dir)
        os.mkdir(test_dir + 'CGG/')
        os.mkdir(test_dir + 'Real/')
        os.mkdir(validation_dir)
        os.mkdir(validation_dir + 'CGG/')
        os.mkdir(validation_dir + 'Real/')


def cut_one_image(input_path, output_path, file_name, step_size):
    img = cv2.imread(input_path + '/' + file_name)
    imgw = img.shape[1]
    imgh = img.shape[0]

    blockcnt = 0
    for px in range(0, imgw-rectWidth, step_size):
        for py in range(0, imgh-rectWidth, step_size):
            #save rect image to file
            cropImg = img[py:py+rectWidth, px:px+rectWidth]
            tName = "%s/%s#%s.bmp" %(output_path, file_name.split('.')[0], str(blockcnt).zfill(4))
            #cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(tName, cropImg)
            #cv2.imwrite(outPathName, img, [int( cv2.IMWRITE_JPEG_QUALITY), 70])
            blockcnt += 1

def cut_image_patches(source_real, source_CG, target_dir, patch_dir, nb_input_images = 1800, 
                 validation_proportion = 0.1, test_proportion = 0.2): 

    make_dirs(target_dir + "/PG2000jpeg75/")
    make_dirs(target_dir + "/PG2000jpeg85/")
    make_dirs(target_dir + "/PG2000jpeg95/")

    make_dirs(patch_dir + "patchPG2000qf75/")
    make_dirs(patch_dir + "patchPG2000qf85/")
    make_dirs(patch_dir + "patchPG2000qf95/")

    image_real = load_images_from_dir(source_real, shuffle = True)
    image_CG = load_images_from_dir(source_CG, shuffle = True)

    nb_train = int(nb_input_images*(1 - validation_proportion - test_proportion))
    nb_test = int(nb_input_images*test_proportion)
    nb_validation = int(nb_input_images*validation_proportion)

    for i in range(nb_train, nb_train + nb_validation):
    	img = cv2.imread(source_real + image_real[i] + ".TIF")
    	cv2.imwrite(target_dir + "/PG2000jpeg95/validation/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    	cv2.imwrite(target_dir + "/PG2000jpeg85/validation/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 85])
    	cv2.imwrite(target_dir + "/PG2000jpeg75/validation/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 75])

        cut_one_image(target_dir + "/PG2000jpeg95/validation/Real/", patch_dir + '/patchPG2000qf95/validation/Real/', image_real[i]+ ".jpg", setpSize)
        cut_one_image(target_dir + "/PG2000jpeg85/validation/Real/", patch_dir + '/patchPG2000qf85/validation/Real/', image_real[i]+ ".jpg", setpSize)
        cut_one_image(target_dir + "/PG2000jpeg75/validation/Real/", patch_dir + '/patchPG2000qf75/validation/Real/', image_real[i]+ ".jpg", setpSize)

        cut_one_image(source_CG, patch_dir + '/patchPG2000qf95/validation/CGG/', image_CG[i]+".jpg", setpSize//10)
        cut_one_image(source_CG, patch_dir + '/patchPG2000qf85/validation/CGG/', image_CG[i]+".jpg", setpSize//10)
        cut_one_image(source_CG, patch_dir + '/patchPG2000qf75/validation/CGG/', image_CG[i]+".jpg", setpSize//10)

    print(str(nb_validation) + ' validation images save to jpeg image')

    for i in range(nb_train):
    	img = cv2.imread(source_real + image_real[i] + ".TIF")
    	cv2.imwrite(target_dir + "/PG2000jpeg95/train/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    	cv2.imwrite(target_dir + "/PG2000jpeg85/train/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 85])
    	cv2.imwrite(target_dir + "/PG2000jpeg75/train/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 75])

        cut_one_image(target_dir + "/PG2000jpeg95/train/Real/", patch_dir + '/patchPG2000qf95/train/Real/', image_real[i]+ ".jpg", setpSize)
        cut_one_image(target_dir + "/PG2000jpeg85/train/Real/", patch_dir + '/patchPG2000qf85/train/Real/', image_real[i]+ ".jpg", setpSize)
        cut_one_image(target_dir + "/PG2000jpeg75/train/Real/", patch_dir + '/patchPG2000qf75/train/Real/', image_real[i]+ ".jpg", setpSize)

        cut_one_image(source_CG, patch_dir + '/patchPG2000qf95/train/CGG/', image_CG[i]+".jpg", setpSize//10)
        cut_one_image(source_CG, patch_dir + '/patchPG2000qf85/train/CGG/', image_CG[i]+".jpg", setpSize//10)
        cut_one_image(source_CG, patch_dir + '/patchPG2000qf75/train/CGG/', image_CG[i]+".jpg", setpSize//10)

    print(str(nb_train) + ' training images save to jpeg image')

    for i in range(nb_train + nb_validation, nb_train + nb_validation + nb_test):
    	img = cv2.imread(source_real + image_real[i] + ".TIF")
    	cv2.imwrite(target_dir + "/PG2000jpeg95/test/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    	cv2.imwrite(target_dir + "/PG2000jpeg85/test/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 85])
    	cv2.imwrite(target_dir + "/PG2000jpeg75/test/Real/" + image_real[i] + ".jpg", img, [int( cv2.IMWRITE_JPEG_QUALITY), 75])

        cut_one_image(target_dir + "/PG2000jpeg95/test/Real/", patch_dir + '/patchPG2000qf95/test/Real/', image_real[i]+ ".jpg", setpSize)
        cut_one_image(target_dir + "/PG2000jpeg85/test/Real/", patch_dir + '/patchPG2000qf85/test/Real/', image_real[i]+ ".jpg", setpSize)
        cut_one_image(target_dir + "/PG2000jpeg75/test/Real/", patch_dir + '/patchPG2000qf75/test/Real/', image_real[i]+ ".jpg", setpSize)

        cut_one_image(source_CG, patch_dir + '/patchPG2000qf95/test/CGG/', image_CG[i]+".jpg", setpSize//10)
        cut_one_image(source_CG, patch_dir + '/patchPG2000qf85/test/CGG/', image_CG[i]+".jpg", setpSize//10)
        cut_one_image(source_CG, patch_dir + '/patchPG2000qf75/test/CGG/', image_CG[i]+".jpg", setpSize//10)

    print(str(nb_test) + ' test images save to jpeg image')

    print("Done.")


if __name__ == '__main__': 

    # Change to the source of real images
    source_real_tif_directory = "/home/forensics/CGPG_experiment/sensor/dataset/PG/"
    # Change to the source of CG images
    source_CG_directory = "/home/forensics/CGPG_experiment/sensor/dataset/CG/"
    # The directory where the database will be saved
    target_dir_test = '/home/forensics/CGPG_experiment/sensor/dataset/'
    patch_dir_test = '/home/forensics/CGPG_experiment/sensor/imagepatches/'

    cut_image_patches(source_real = source_real_tif_directory, 
              source_CG = source_CG_directory,
              target_dir = target_dir_test,
              patch_dir = patch_dir_test,  
              nb_input_images = 1800,
              validation_proportion = 0.1,
              test_proportion = 0.4)
    