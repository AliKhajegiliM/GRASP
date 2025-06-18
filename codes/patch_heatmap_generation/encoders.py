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
torch.manual_seed(256)
import torch.nn as nn

def load_state_dict_with_prefix(model, state_dict, prefix='module.'):
    
    # Remove the specified prefix from the keys in the state_dict
    modified_state_dict = {key.replace(prefix, ''): value for key, value in state_dict.items()}

    # Load the modified state_dict into the model
    model.load_state_dict(modified_state_dict, strict=True)
    return model

def pre_trained_model(model_name, device, custom_weights_path=None):
    # Load pre-trained models
    #resnet50
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True).to(device)
        if custom_weights_path != None:
            model.fc = nn.Linear(2048, num_classes)
            weights = torch.load(custom_weights_path, map_location=torch.device(device))
            if 'net' in weights.keys():
                weights = weights['net']
            elif 'state_dict' in weights.keys():
                weights = weights['state_dict']
                try:
                    model.load_state_dict(weights)
                except:
                    model = load_state_dict_with_prefix(model, weights, prefix='module.model.')
        model.fc = nn.Sequential()
        #print(model)
    #resnet50_cds
    elif model_name == 'resnet50_cds':
        import sys
        sys.path.append('/projects/ovcare/classification/Ali/Search_Engine/cross_domain_adaptation/codes/CDS/CDS_pretraining')
        import models
        inc = 2048
        model = models.__dict__['resnet50'](pretrained=True, low_dim=512)
        #model = nn.DataParallel(model)
        if custom_weights_path != None:
            weights = torch.load(custom_weights_path, map_location=torch.device(device))
            weights = weights['net']
            model.load_state_dict(weights, strict=False)
        model.fc = nn.Sequential()
        #print(model)
    #convnext_base
    elif model_name == 'convnext_base':
        model = torchvision.models.convnext_small(pretrained=True)
        if custom_weights_path != None:
            weights = torch.load(custom_weights_path, map_location=torch.device(device))
            #weights = weights['state_dict']
            model.load_state_dict(weights, strict=False)
        model.classifier[2] = nn.Sequential()
    #densenet121
    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True).to(device)
        model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
        model.classifier = nn.Sequential()
    #vit
    elif model_name == 'vit' or model_name == 'vit_b16':
        #import torch.nn as nn
        model = torchvision.models.vit_b_16(pretrained=True).to(device)
        if custom_weights_path != None:
            model.heads = nn.Linear(768, num_classes)
            weights = torch.load(custom_weights_path, map_location=torch.device(device))
            if 'net' in weights.keys():
                weights = weights['net']
            elif 'state_dict' in weights.keys():
                weights = weights['state_dict']
                try:
                    model.load_state_dict(weights)
                except:
                    model = load_state_dict_with_prefix(model, weights, prefix='module.')
        model.heads = nn.Sequential()
        #print(model)
    #swin_b
    elif model_name == 'swin' or model_name == 'swin_b':
        model = torchvision.models.swin_b(pretrained=True).to(device)
        if custom_weights_path != None:
            model.head = nn.Linear(1024, num_classes)
            weights = torch.load(custom_weights_path, map_location=torch.device(device))
            if 'net' in weights.keys():
                weights = weights['net']
            elif 'state_dict' in weights.keys():
                weights = weights['state_dict']
                try:
                    model.load_state_dict(weights)
                except:
                    model = load_state_dict_with_prefix(model, weights, prefix='module.')
        model.head = nn.Sequential()
        #print(model)
    #swin_t
    elif model_name == 'swin_t':
        model = torchvision.models.swin_t(pretrained=True).to(device)
        if custom_weights_path != None:
            model.head = nn.Linear(768, num_classes)
            weights = torch.load(custom_weights_path, map_location=torch.device(device))
            if 'net' in weights.keys():
                weights = weights['net']
            elif 'state_dict' in weights.keys():
                weights = weights['state_dict']
                try:
                    model.load_state_dict(weights)
                except:
                    model = load_state_dict_with_prefix(model, weights, prefix='module.')
        model.head = nn.Sequential()
    #PLIP
    elif model_name == 'PLIP':
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        class my_model(nn.Module):
            def __init__(self, model):
                super(my_model, self).__init__()
                self.model = model
            def forward(self, img):
                return self.model.get_image_features(pixel_values = img)

        def plip(pretrained=True, low_dim=512):
            """Constructs a ViT-b16 model.

            Args:
                pretrained (bool): If True, returns a model pre-trained on ImageNet
            """
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
            processor = AutoProcessor.from_pretrained("vinid/plip")
            model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
            model = my_model(model)
            return model
        processor = AutoProcessor.from_pretrained("vinid/plip")
        model = plip()
        if custom_weights_path != None:
            weights = torch.load(custom_weights_path, map_location=torch.device(device))
            weights = weights['net']
            model.load_state_dict(weights)
    #CTransPath
    elif model_name == 'CTransPath':
        from timm.models.layers.helpers import to_2tuple
        import timm
        #import torch.nn as nn
        
        class ConvStem(nn.Module):
            def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
                super().__init__()

                assert patch_size == 4
                assert embed_dim % 8 == 0

                img_size = to_2tuple(img_size)
                patch_size = to_2tuple(patch_size)
                self.img_size = img_size
                self.patch_size = patch_size
                self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
                self.num_patches = self.grid_size[0] * self.grid_size[1]
                self.flatten = flatten


                stem = []
                input_dim, output_dim = 3, embed_dim // 8
                for l in range(2):
                    stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
                    stem.append(nn.BatchNorm2d(output_dim))
                    stem.append(nn.ReLU(inplace=True))
                    input_dim = output_dim
                    output_dim *= 2
                stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
                self.proj = nn.Sequential(*stem)

                self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

            def forward(self, x):
                B, C, H, W = x.shape
                assert H == self.img_size[0] and W == self.img_size[1], \
                    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                x = self.proj(x)
                if self.flatten:
                    x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                x = self.norm(x)
                return x
        model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        model.head = nn.Identity()
        td = torch.load('/projects/ovcare/classification/Feature_Extractors_Weights/ctranspath/ctranspath.pth')
        model.load_state_dict(td['model'], strict=False)
    #Phikon
    elif model_name == 'Phikon':
        from transformers import AutoImageProcessor, AutoModel
        class my_model(nn.Module):
            def __init__(self, model):
                super(my_model, self).__init__()
                self.model = model
            def forward(self, img):
                return self.model(img)[1]
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = AutoModel.from_pretrained("owkin/phikon")
        model = my_model(model)
    #Lunit-Dino 
    elif model_name == 'Lunit-Dino':
        from timm.models.vision_transformer import VisionTransformer
        def get_pretrained_url(key):
            URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
            model_zoo_registry = {
                "DINO_p16": "dino_vit_small_patch16_ep200.torch",
                "DINO_p8": "dino_vit_small_patch8_ep200.torch",
            }
            pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
            return pretrained_url

        def vit_small(pretrained, progress, key, **kwargs):
            patch_size = kwargs.get("patch_size", 16)
            model = VisionTransformer(
                img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
            )
            if pretrained:
                pretrained_url = get_pretrained_url(key)
                verbose = model.load_state_dict(
                    torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
                )
                print(verbose)
            return model
        model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
    elif model_name == 'KimiaNet':
        class my_model(nn.Module):
            def __init__(self, model):
                super(my_model, self).__init__()
                self.model = model
            def forward(self, img):
                _, out = self.model(img)
                return out
        class fully_connected(nn.Module):
            def __init__(self, model, num_ftrs, num_classes):
                super(fully_connected, self).__init__()
                self.model = model
                self.fc_4 = nn.Linear(num_ftrs,num_classes)
            
            def forward(self, x):
                x = self.model(x)
                x = torch.flatten(x, 1)
                out_1 = x
                out_3 = self.fc_4(x)
                return  out_1, out_3
        
        model = torchvision.models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
        num_ftrs = model.classifier.in_features
        model = fully_connected(model.features, num_ftrs, 30)
        model = model.to(device)
        model = nn.DataParallel(model)
        params_to_update = []
        model.load_state_dict(torch.load('/projects/ovcare/classification/Ali/Ovarian_project/Pytorch_Codes/KimiaNet/KimiaNetPyTorchWeights.pth',map_location=torch.device('cpu')))
        model.module.fc_4 = nn.Sequential()
        model = my_model(model)

    else:
        print('Error: unsupported model')
        exit()
    return model