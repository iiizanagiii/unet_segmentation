import time
st = time.time()

import math
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import yaml

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import segmentation_model

take_time = time.time()-st
print("Time for importing libraries: ",math.floor(take_time),"seconds" )
dir = os.getcwd()

#Class for custom datasets
class SegmentationDataset(Dataset):

  def __init__(self,df):
    self.df = df
    # self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self,idx):
    row = self.df.iloc[idx]
    opt = parse_opt()
    config = load_config(opt)

    image_path = dir + config['train'] + row.images
    mask_path = dir + config['test'] + row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims(mask,axis=-1)

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask) / 255.0)

    return image,mask
  
  # ----------------------------------------------------------------

#Function for taining
def train(model, device, data_loader, optimizer, epoch):
  size = len(data_loader.dataset)
  model.train()
  total_loss = 0.0

  for  batch , (data, target) in enumerate(data_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    logits,loss = model(data,target)
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
    
    if batch % 100 == 0 :
      loss, current = loss.item(), (batch + 1) * len(data)
      print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
        #     epoch, batch * len(data), len(data_loader.dataset),
        #     100. * batch / len(data_loader), loss.item()))

#   return total_loss / len(data_loader)

#function for validation
def valid(model, device, data_loader):

  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for data, target in (data_loader):
      data = data.to(device)
      target = target.to(device)
      logits,loss = model(data,target)


      total_loss+=loss.item()
      
    print('\n Valid Loss: {:.6f}'.format(total_loss))

#   return total_loss / len(data_loader)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, help='number of epochs to train for')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for optimizer')
    parser.add_argument('--batch', default=1, type=int, help='batch size for data loader')
    parser.add_argument('--csv_file', default='./tablets_segmentation/image_mask_pairs.csv', type=str, help='csv file of images and masks')
    parser.add_argument('--device', default='', type=str, help='Device "Cuda or CPU')
    parser.add_argument('--img_size', default=512, type=int, help='image size for data loader')
    parser.add_argument('--encoder', default='timm-efficientnet-b0', type=str, help='Encoder for backbone')
    parser.add_argument('--weights', default='imagenet', type=str, help='Default weights is imagenet')
    # parser.add_argument('--image_location',default='/tablets_segmentation/Training_Images/',help='Location for original training images',type=str)
    # parser.add_argument('--mask_location',default='/tablets_segmentation/Ground_Truth/',help='Location for ground truth images',type=str)
    parser.add_argument('--save_model', default='/model/', type=str, help='saving the model in the directory')
    parser.add_argument('--data', default='data.yaml', type=str, help='YAML configuration file')

    return parser.parse_args()
    
    
def load_config(args):
  with open(args.data) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  
  return config
#----------------------------------------------------------------

def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir = os.getcwd()
    df = pd.read_csv(dir + opt.csv_file)
    train_df, valid_df = train_test_split(df, test_size = 0.20, random_state = 42)
    print('Total number of train images',len(train_df))
    print('Total number of valid images',len(valid_df))
    
    train_set = SegmentationDataset(train_df)
    valid_set = SegmentationDataset(valid_df)
    
    print(f"Size of Trainset : {len(train_set)}")
    print(f"Size of Validset : {len(valid_set)}")
    
    trainloader = DataLoader(train_set, batch_size=opt.batch, shuffle=True)
    validloader = DataLoader(valid_set, batch_size=opt.batch, shuffle=False)
    
    model = segmentation_model.SegmentationModel(opt.encoder,opt.weights)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_valid_loss = np.Inf
    
    for epoch in range(opt.epochs):
        print(f'Epoch {epoch+1}\n---------------')
        train(model, device, trainloader, optimizer, epoch)
        valid(model, device, validloader)
        
        if opt.save_model:
            torch.save(model.state_dict(),'best.pt')
            print('[+] model saved')
            
            
def run(**kwargs):
    opt = parse_opt()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == '__main__':
  run(
      epochs = 4,
      lr = 0.0001,
      batch=1,
      csv_file='./tablets_segmentation/image_mask_pairs.csv',
      img_size='512',
      encoder='timm-efficientnet-b0',
      weights='imagenet',
      save_model='/model/',
      data='data.yaml'
    
  )
#----------------------------------------------------------------------  
# args = parser.parse_args()
# print(args)

# #Setup configuration
# df = pd.read_csv( dir +args.csv_file)

# train_df, valid_df = train_test_split(df, test_size = 0.20, random_state = 42)

# print("Total number of train images",len(train_df))
# print("Total number of valid images",len(valid_df))


# #Create custom dataset
# trainset = SegmentationDataset(train_df)#,get_train_augs(args.img_size))
# validset = SegmentationDataset(valid_df)#,get_valid_augs(args.img_size))

# print(f"Size of Trainset : {len(trainset)}")
# print(f"Size of Validset : {len(validset)}")

# #Load dataset into batches
# trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
# validloader = DataLoader(validset, batch_size=args.batch, shuffle=False)

# #Segmentation Model
# model = segmentation_model.SegmentationModel(args.encoder,args.weights)
# model.to(args.device)

# #Train Mode

# optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# # best_valid_loss = np.Inf


# best_valid_loss = np.Inf

# for i in range(args.epochs):
#   train_loss = train_fn(trainloader,model,optimizer)
#   valid_loss = valid_fn(validloader,model)

#   if valid_loss < best_valid_loss:
#     torch.save(model.state_dict(),'best.pt')
#     print("SAVED MODEL")
#     best_valid_loss = valid_loss

#   print(f"Epoch: {i+1} Train loss : {train_loss}  Valid loss: {valid_loss}")
    