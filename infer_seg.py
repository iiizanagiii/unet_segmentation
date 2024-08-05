import segmentation_model
import argparse
import torch
import os
from PIL import Image
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = 'timm-efficientnet-b0'
weights = 'imagenet'
model = segmentation_model.SegmentationModel(encoder,weights)
model.to(device)
model_name = '../best.pt'
model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
input_images = './good/'

all_image_paths = os.listdir(input_images)
for i, image_path in enumerate(all_image_paths):
    img = cv2.imread(os.path.join(input_images, image_path))
    copy_img = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = np.transpose(img, (2, 0, 1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0
    logits_mask = model(image.to(device).unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)
    ori_image = pred_mask.detach().cpu().numpy()
    ori_image = ori_image.squeeze() # Remove the single-dimensional entries
    ori_image = (ori_image * 255).astype(np.uint8)
    
    # Convert the mask to a binary mask (thresholding)
    ret, binary_mask = cv2.threshold(ori_image, 127, 255, cv2.THRESH_BINARY)
    
    # Invert the binary mask to get the area to be removed
    inverse_mask = cv2.bitwise_not(binary_mask)
    bit_x = cv2.bitwise_and(copy_img,copy_img,mask = inverse_mask)
    bit_x = cv2.resize(bit_x,(255,255))
    cv2.imshow('image',bit_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Successfully Infered")
