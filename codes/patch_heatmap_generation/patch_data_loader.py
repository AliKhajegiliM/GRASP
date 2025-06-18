import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp
import argparse
import numpy as np
import cv2
import os
import glob
from PIL import Image
import h5py


def get_patch_path(patch_location, mags):
        extentions = ['png','jpg']
        patch_path = {}
        for mag in mags:
            patch_path[mag + 'x'] = []
            for ext in extentions:
                path_wildcard = os.path.join(patch_location, '**/'+ mag + '/**', '*.' + ext)
                patch_path[mag + 'x'].extend(glob.glob(path_wildcard, recursive=True))
            patch_path[mag + 'x'] = sorted(patch_path[mag + 'x'])

        if len(patch_path)==0:
            raise ValueError("Are you kidding me! This directory is empty")
        
        for mag in list(patch_path.keys()):
            if len(patch_path[mag]) != len(patch_path[list(patch_path.keys())[0]]):
                print(len(patch_path[mag]))
                raise ValueError("GOD Bless you bro! You are indeeed a genious!! Go check your patches, you have not extracted equal numbers of patches from each magnification")
        
        slide = {}
        for mag in mags:
            for item in patch_path[mag+'x']:
                slide_name = item.split('/')[-4]
                
                if slide_name not in slide:
                    slide[slide_name] = {mag+'x': [item]}
                else:
                    if mag+'x' not in slide[slide_name]:
                        slide[slide_name][mag+'x'] = [item]
                    else:
                        slide[slide_name][mag+'x'].append(item)

        return slide

class data_loader(Dataset):
    def __init__(self, slide_patches, image_transform, mags_list=['20x'], transform=True, center_crop=True):
        self.patch_path=slide_patches
        self.transform=transform
        self.mags_list=mags_list
        self.center_crop=center_crop
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.patch_path[self.mags_list[0]])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patches_tensor = {}
        patches_fr = {}
        for mag in self.mags_list:
            patches_tensor[mag] = []
            patches_fr[mag] = []
            path_to_patch = self.patch_path[mag][idx]
            subtype = path_to_patch.split('/')[-5]
            patch_coords = path_to_patch.split('/')[-1].split('.')[0].split('_')
            # Load the image as a PIL image
            img_pil = Image.open(path_to_patch).convert('RGB')
            # mask:
            img = np.array(img_pil)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = cv2.threshold(gray, 0, 255, 1)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours==():
                tape_contour = max(contours, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(tape_contour)

                #print(x,y,w,h)
                e = 5 
                if x == 0 and y==0:
                    u = min([w,h])+e
                    #print(u)
                    poc = img[u:u+e, u:u+e] 
                else:
                    u = max([w,h]) - min([w,h]) - e
                    #print(u)
                    poc = img[u-e:u, u-e:u] 
                #pixel of choice
                pixel_set = np.mean(np.mean(poc,0),0).astype(np.uint8)
                img[mask==255] = pixel_set
            
            # Define the patch size
            if self.center_crop:
                patch_size = img.shape[0]
            else:
                patch_size = 224
            # Split the image into patches of size patch_size x patch_size
            patches = []
            import matplotlib.pyplot as plt
            patches_coord = [] # (x,y)
            patch_fr = [] #foreground ratio
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            otsu = Image.fromarray(otsu)
            for i in range(0, img.shape[0], patch_size):
                for j in range(0, img.shape[1], patch_size):
                    patch = img_pil.crop((i, j, i+patch_size, j+patch_size)) # RGB np array, this should be modified for the original run of TCGA
                    patches.append(patch)
                    patches_coord.append(torch.tensor([i + int(patch_coords[0]), j + int(patch_coords[1])]))
                    otsu_patch = np.array(otsu.crop((i, j, i+patch_size, j+patch_size)))
                    fr = sum(sum(otsu_patch==0))/otsu_patch.shape[0]/otsu_patch.shape[1]
                    patch_fr.append(torch.tensor([fr]))
                    

            # Convert each patch to a PyTorch tensor
            patch_tensors = [torch.tensor(np.array(patch)) for patch in patches]
            # Stack the tensors into a single tensor
            patches_tensor[mag] = torch.stack(patch_tensors)
            patches_coord = torch.stack(patches_coord)
            patches_fr[mag] = torch.stack(patch_fr)
            if self.transform:
                # Apply data augmentation if requested
                transformed_patches = [self.image_transform(patch) for patch in patches]
                patches_tensor[mag] = torch.stack(transformed_patches)
        
        patches_tensor = torch.stack(list(patches_tensor.values()))
        #print(patches_tensor.shape)
        patches_fr = torch.stack(list(patches_fr.values()))
        return(patches_tensor, subtype, patches_coord, patches_fr)