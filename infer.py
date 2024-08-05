import segmentation_model
import argparse
import torch
import os
from PIL import Image
import cv2
import numpy as np


def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 0.5 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def tensor_to_numpy(copy_img,pred_mask,image_path,out_dir):
    image_np = pred_mask.detach().cpu().numpy()
    image_np = image_np.squeeze()  # Remove the single-dimensional entries
    image_np = (image_np * 255).astype(np.uint8)

    # Step 3: Convert NumPy array to OpenCV image
    # opencv_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)


    # Convert the mask to a binary mask (thresholding)
    ret, binary_mask = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)

    # Invert the binary mask to get the area to be removed
    inverse_mask = cv2.bitwise_not(binary_mask)

    bit_x = cv2.bitwise_and(copy_img,copy_img,mask = inverse_mask)

    cv2.imwrite(os.path.join(out_dir, image_path), bit_x)
    # cv2.imwrite(os.path.join(out_dir, image_path), opencv_image)

    # final_image = image_overlay(copy_img, opencv_image)
    # cv2.imwrite(os.path.join(out_dir, image_path), final_image)



out_dir = 'inference_me'
os.makedirs(out_dir, exist_ok=True)


parser = argparse.ArgumentParser()

parser.add_argument(
    '--device',
    default='cpu',
    help='Device "Cuda or CPU',
    type=str
)




parser.add_argument(
    '--encoder',
    default='timm-efficientnet-b0',
    help='Encoder to use',
    type=str 
)

parser.add_argument(
    '--weights',
    default='imagenet',
    help='Encoder to use',
    type=str
)

parser.add_argument('-i', '--input', help='path to input dir')


args = parser.parse_args()

encoder = args.encoder
weights = args.weights


model = segmentation_model.SegmentationModel(encoder,weights)
model.to(args.device)

model_name = '../best.pt'


model.load_state_dict(torch.load(model_name, map_location=torch.device(args.device)))

all_image_paths = os.listdir(args.input)
for i, image_path in enumerate(all_image_paths):
    img = cv2.imread(os.path.join(args.input, image_path))
    copy_img = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = np.transpose(img, (2, 0, 1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0
    logits_mask = model(image.to(args.device).unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)
    tensor_to_numpy(copy_img,pred_mask, image_path,out_dir)
    print(image_path)

print("Successfully Infer")
