import imageio
import numpy as np
import imgaug as ia 
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt
import cv2
from imgaug.augmenters.rigid_ import Rigid_Deform
import os

import yaml 

with open('./config.yaml') as config_file:
    config =  yaml.safe_load(config_file)

# Create folders for augmented images
augmented_image_folder = config['augment_image_dir'] #'./data_augment/white-oval-tabs/images'
augmented_mask_folder =config['augment_mask_dir'] #'./data_augment/white-oval-tabs/mask'
augmented_uni_mask_folder = config['augment_u_mask_dir']


os.makedirs(augmented_image_folder, exist_ok=True)
os.makedirs(augmented_mask_folder, exist_ok=True)
# os.makedirs(augmented_uni_mask_folder, exist_ok=True)

# Load an example image and segmentation mask
path_img = config['original_image_dir'] #'./data_original/white-oval-tabs/images'
path_mask =  config['original_mask_dir'] #'./data_original/white-oval-tabs/mask'

#function to sort the images
def sort_files(image_folder):
  # Get a list of image filenames in the folder
  image_files = os.listdir(image_folder)
  # Sort the filenames in serial order
  sorted_files = sorted(image_files, key=lambda x: str(x.split('.')[0]))
  return sorted_files

def top_bottom_hat(image, kernel_size = (35, 35), morph_type = cv2.MORPH_ELLIPSE):
    # pass the image to contrast enhancement
    image_gray = image.copy()
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(morph_type, kernel_size)
    tophat = cv2.morphologyEx(image_gray, morph_type, kernel)
    blackhat = cv2.morphologyEx(image_gray, morph_type, kernel)
    result_image = image_gray + tophat - blackhat
    return result_image

#sort the filenames
sorted_img_path = sort_files(path_img)
sorted_mask_path = sort_files(path_mask)

# Loop through all the files in the folder
count = 0
IMAGE_SIZE = 512 # 256
for i in range(len(os.listdir(path_img))-1):
    image = cv2.imread(path_img+'/'+str(sorted_img_path[i]))
    image = cv2.resize(image,(IMAGE_SIZE, IMAGE_SIZE))
    mask = cv2.imread(path_mask+'/'+str(sorted_mask_path[i])) 
    mask = cv2.resize(mask,(IMAGE_SIZE, IMAGE_SIZE))

    # mask = cv2.bitwise_not(mask)

    segmap = SegmentationMapsOnImage(mask, shape=mask.shape)

    # Define the augmentations
    augmentations = [
        (iaa.Affine(rotate=(-25, 25)), 'rotate'),
        (iaa.GaussianBlur(sigma=(0, 3.0)), 'blur'),
        (iaa.ContrastNormalization((0.5, 2.0)), 'contrast'),
        # (iaa.ElasticTransformation(alpha=10, sigma=3), 'elastic'),
        (iaa.Fliplr(p=1.0), 'fliplr'),
        (iaa.Flipud(p=1.0), 'flipud'),
        # (iaa.Crop(percent=(0, 0.3)), 'crop')
    ]

    for j in range(10): # n-times

        distance = 15  # Example distance value
        points = 10  # Example points value

        # deformer = Rigid_Deform(distance, points)
        # result,mask_result = deformer.rigid_deformation(image,mask)

        # tbh = top_bottom_hat(image)
        
        # Apply individual augmentations to the image and mask
        augmented_images = []
        augmented_images_mask = []
        for augmentation,name in augmentations:
            seq = iaa.Sequential([augmentation])
            augmented_image, augmented_mask = seq(image=image, segmentation_maps=segmap)
            augmented_mask = augmented_mask.get_arr()
            augmented_images.append((augmented_image,name))
            augmented_images_mask.append((augmented_mask,name))

        # augmented_images.append((result,'rigid'))
        # augmented_images.append((tbh,'tbh'))
        # augmented_images_mask.append((mask_result,'rigid'))
        # augmented_images_mask.append((mask_result,'tbh'))
        augmented_images.append((image,'original'))
        augmented_images_mask.append((mask,'original'))
        # Save the augmented images
        for k, (augmented_image,name) in enumerate(augmented_images):
            count+=1
            cv2.imwrite(f'{augmented_image_folder}/{name}_{j+1}_{i}.jpg', augmented_image)
            cv2.imwrite(f'{augmented_mask_folder}/{name}_{j+1}_{i}.jpg', augmented_images_mask[k][0])

# Note: The augmented mask images may need further processing or adjustment depending on the task.
print("total number of images augmentation is",count)