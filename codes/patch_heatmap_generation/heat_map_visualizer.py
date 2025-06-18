import sys
import os
sys.path.append("/projects/ovcare/classification/Ali/Bladder_project/codes")
sys.path.append("/projects/ovcare/classification/Ali/Bladder_project/codes/Benchmarking")
sys.path.append('/projects/ovcare/classification/Ali/Bladder_project/codes/patch_heatmap_generation')
import torch
from torchvision import transforms
import torch.nn as nn
from encoders import pre_trained_model as model_loader
from patch_data_loader import get_patch_path, data_loader
import numpy as np
from scipy import sparse
import dgl
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.metrics import accuracy_score as acc
import argparse

parser = argparse.ArgumentParser(description='Reporting Cleaned Results for each model')
parser.add_argument("--model", nargs='+', type=str, required = True, help="<<<< --models GRASP>>>> ")
parser.add_argument("--encoder", nargs='+', type=str, required = True, help="<<<< --models vit>>>> ")
parser.add_argument("--hidden_layers", nargs='+', type=int,  default = [256, 128], 
        help="<<<< --hidden_layers [256 128] >>>> keep it consistent!")
parser.add_argument("--mags", nargs='+', type=str, required = True, help="<<<< --models 5 10 20 >>>> ")
parser.add_argument("--batch_size", nargs='+', type=int, required=True, default = [1],  help="<<<< --batch_size 16 >>>> ")
parser.add_argument("--num_folds", nargs='+', type=int, required=True, default = [3],  help="<<<< --batch_size 16 >>>> ")
parser.add_argument("--feat_size", nargs='+', type=int, required=True, default = [768],  help="<<<< --num_folds 3 >>>> ")
parser.add_argument("--lr", nargs='+', type=float, default = [0.001], help="<<<< --lr 0.001 >>>> ")
parser.add_argument("--weight_decay", nargs='+', type=float, default = [0.0001],  help="<<<< --weight_decay 0.0001 >>>> ")
parser.add_argument("--epochs", nargs='+', type=int, default = [50],  help="<<<< --epochs 50 >>>> ")
parser.add_argument("--classes", nargs='+', type=str, required = True, 
        help="<<<< --classes UCC:0 MicroP:1 >>>> ")
parser.add_argument("--crop_size", nargs='+', type=int, required=True, default = [0],  help="<<<< --num_folds 3 >>>> ")
parser.add_argument('--path_to_checkpoints', type=str, required=True, help='path to the model checkpoints')
parser.add_argument('--path_to_outputs', type=str, required=True, help='path to the model outputs')
parser.add_argument('--patch_location', type=str, required=True, help='path to patches')
parser.add_argument('--path_to_save_heatmaps', type=str, required=True, help='path to the data fold json file')


args = parser.parse_args()

mil_model_name = args.model[0]
encoder_name = args.encoder[0]
hidden_layers = args.hidden_layers
mags = args.mags
batch_size = args.batch_size[0]
num_folds = args.num_folds[0]
in_size = args.feat_size[0]
lr = args.lr[0]
weight_decay = args.weight_decay[0]
epochs = args.epochs[0]
path_to_outputs = args.path_to_outputs
model_checkpoint_path = args.path_to_checkpoints
patch_location = args.patch_location
classes = args.classes
path_to_save = args.path_to_save_heatmaps
crop_size = args.crop_size[0]
if crop_size != 0:
    center_crop_flag = True
torch.manual_seed(256)
random_seeds = torch.randint(0, 10000, (10,)).numpy()
multi_mag = ["GRASP", "ZoomMIL", "H2MIL", "HiGT"]
single_mag = ["DeepMIL", "VarMIL", "TransMIL", "Clam_SB", "Clam_MB","PatchGCN", "DGCN"]


# patch_location = '/projects/ovcare/classification/Ali/Heram/Dataset/Bladder_dataset/patches/Mix/'
# model_checkpoint_path = '/projects/ovcare/classification/Ali/Heram/codes/Bladder_codes/new_benchmark/results/vit/checkpoint_TransMIL_fold-1_5x_S278_B1_LR0.001_WD0.01_E20.pt'
# path_to_save = '/projects/ovcare/classification/Ali/Heram/codes/Bladder_codes/new_benchmark/heatmaps/'
# mags = ['5', '10', '20']
# mil_model_name = 'DeepMIL'
# encoder_name = 'vit'
# crop_size = 224
# in_size = 768
# hidden_layers = [256, 128]
# batch_size = 1
# classes = ['UCC:0', 'MicroP:1']

label_dict = {}
for item in classes:
    label = item.split(':')
    label_dict[label[0]] = int(label[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #

slides = get_patch_path(patch_location, mags)
print(len(slides))

if center_crop_flag:
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
else:
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

def center_crop(image, target_size):
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate starting point for cropping
    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)

    # Perform center crop
    cropped_image = image[start_h:start_h + target_h, start_w:start_w + target_w]

    return cropped_image

def slide_picker(slides, sample_per_slide=5, random_seed=43):
    np.random.seed(random_seed)
    # Randomly select samples from each slide
    selected_samples = {}
    for slide in slides:
        mags = list(slides[slide].keys())
        data = slides[slide][mags[0]]
        if sample_per_slide > len(data):
            raise ValueError("Number of samples per slide exceeds the total number of patches.")
        selected_samples[slide] = {}

        random_indices = np.random.choice(len(data), size=sample_per_slide, replace=False)
        print(random_indices, len(data))
        for mag in mags:
            print(mag)
            selected_samples[slide][mag] = list(np.array(slides[slide][mag])[random_indices])

    return selected_samples
    

def best_seed(result_dict, random_seeds, criterion = 'bacc'):
    x = result_dict[criterion]
    best_seed = np.argmax(np.mean(x, axis=1))
    the_best = {}
    for key in result_dict.keys():
        the_best[key] = result_dict[key][best_seed]
    #print(the_best)
    return random_seeds[best_seed], the_best

results ={}
results[mil_model_name] = {}
for seed in random_seeds:
    results[mil_model_name][seed] = {'acc':[], 'bacc':[], 'auc':[], 'f1':[], 'time':[], 'recall':[], 'precision':[]}
    for fold in range(num_folds):
        split_name = 'fold-' + str(fold+1)
        if mil_model_name in multi_mag:
            if mil_model_name == 'ZoomMIL':
                hidden_layers = [256, 512]
            model_details = mil_model_name + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
            structure = 'output_' + model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'
        elif mil_model_name in single_mag:
            structure = 'output_' + mil_model_name + '_' + split_name + '_'  + mags[0] + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay)+ '_' + 'E' + str(epochs) + '.pt'
        data = torch.load(path_to_outputs + structure)
        accuracy = acc(data['labels'], data['preds'])
        balanced = bas(data['labels'], data['preds'])
        #print(area_uc)
        results[mil_model_name][seed]['acc'].append(accuracy)
        results[mil_model_name][seed]['bacc'].append(balanced)

avg = {'acc':[], 'bacc':[]}
stdv = {'acc':[], 'bacc':[]}
#print(results[model])
tmp = {'acc':[], 'bacc':[]}
for seed in random_seeds:
    tmp['acc'].append(results[mil_model_name][seed]['acc'])
    tmp['bacc'].append(results[mil_model_name][seed]['bacc'])
tmp['acc'] = np.array(tmp['acc'])
tmp['bacc'] = np.array(tmp['bacc'])
best_seed_value, _ = best_seed(tmp, criterion='bacc', random_seeds=random_seeds)
print(mil_model_name,'--best_seed: ', best_seed_value)
if mil_model_name in multi_mag:
    if mil_model_name == 'ZoomMIL':
        hidden_layers = [256, 512]
    model_details = mil_model_name + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
    ch_structure = 'checkpoint_' + model_details + '_' + split_name + '_' + 'S' + str(best_seed_value) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'
elif mil_model_name in single_mag:
    ch_structure = 'checkpoint_' + mil_model_name + '_' + split_name + '_'  + mags[0] + '_' + 'S' + str(best_seed_value) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay)+ '_' + 'E' + str(epochs) + '.pt'
mil_weights = torch.load(model_checkpoint_path + ch_structure, map_location=torch.device(device))

mil_model = mil_weights["model"]
print(mil_model)
encoder_model = model_loader(encoder_name, device=device, custom_weights_path=None)
encoder_model = encoder_model.to(device)
slides = slide_picker(slides, sample_per_slide=1, random_seed=43)
#print(slides)
mags = [i+'x' for i in mags]
from PIL import Image
from scipy.ndimage import gaussian_filter
for slide in slides:
    torch.cuda.empty_cache()
    print('....................working on slide: '+ slide, flush = True)
    raw_data = data_loader(slides[slide], image_transform = image_transform, mags_list = mags, transform=True, center_crop = center_crop)
    loaded_data = torch.utils.data.DataLoader(raw_data, batch_size=1, shuffle=False, num_workers=2)
    patch_ind = -1
    for image, subtype, patch_coords, patch_frs in loaded_data:
        patch_ind += 1
        image = image.to(device)
        image = image.squeeze()
        #print(image.shape)
        label = torch.tensor(label_dict[subtype[0]]).to(device)
        image.requires_grad_()
        features = encoder_model(image)
        #print(features.shape)
        n = 1
        n_mag = len(mags)
        A = np.zeros((n_mag*n,n_mag*n))
        for block_row in range(n_mag):
            for block_col in range(n_mag):
                if block_row == block_col:
                    A[block_row*n:(block_row+1)*n, block_col*n:(block_col+1)*n] = np.ones((n,n))
                if abs(block_row-block_col)==1:
                    A[block_row*n:(block_row+1)*n, block_col*n:(block_col+1)*n] = np.diag(np.ones((1,n)).reshape(n,))
        sA = sparse.csr_matrix(A)
        g = dgl.from_scipy(sA)
        g = g.to(device)
        g.ndata['x'] = features
        #print(g)
        if mil_model_name in ['GRASP', 'ZoomMIL', 'H2MIL']:
            output, attention = mil_model(g.to(device), g.ndata['x'].to(device))
        elif mil_model_name in ['DeepMIL', 'TransMIL', 'VarMIL', 'Clam_SB', 'Clam_MB']:
            output, attention = mil_model(g.ndata['x'][2].reshape(1,1,g.ndata['x'].shape[1]))
        #print(output)
        output_idx = output.argmax()
        output_max = output[0, output_idx]
        output_max.backward()
        #saliency=images.grad.data.abs()
        saliency, _ = torch.max(image.grad.data.abs(),dim=1)
        saliency = saliency.cpu()
        #print(saliency.shape)
        org_img = {}
        mag_ind = 0
        for mag in mags:
            #print(mag)
            #u_img = cv2.imread(slides[slide][mag][patch_ind], cv2.COLOR_BGR2RGB)
            image_path = slides[slide][mag][patch_ind]
            coords = image_path.split('/')[-1].split('.')[0]
            pil_image = Image.open(image_path).convert('RGB')
            if mag_ind == 0:
                print('patch coordination: ', coords)

            # Convert PIL image to NumPy array
            u_img = np.array(pil_image)
            u_img = center_crop(u_img, [crop_size, crop_size])
            ht = np.array(saliency[mag_ind].reshape(crop_size,crop_size))
            mag_ind += 1
            
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable

            # ...

            # Inside your loop:

            # Assuming ht is in the range [0, 1]
            maxval = 255
            ht_normalized_ = ht / np.max(ht)
            im_bin = (ht_normalized_ >0.15) * maxval
            binary_image = im_bin.astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

            # Apply Gaussian blur to the dilated image
            ht = cv2.GaussianBlur(dilated_image.astype(float), (15, 15), 0)
            ht_normalized = ht / np.max(ht)
            beta = 0.5
            ht_normalized = beta*ht_normalized + (1-beta)*ht_normalized_
            # Apply your custom colormap using matplotlib
            cmap = plt.get_cmap('hot')  # You can change 'hot' to other available colormaps
            norm = Normalize(vmin=0, vmax=1)  # Normalize to [0, 1]

            # Use ScalarMappable to map normalized values to colors
            sm = ScalarMappable(cmap=cmap, norm=norm)
            colored_ht = sm.to_rgba(ht_normalized)

            # Display the histogram of the flattened matrix
            plt.figure()
            plt.hist(ht.flatten(), bins=30, color='blue', alpha=0.7)
            plt.title('Histogram of Flattened Matrix')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.show()

            # Apply the heatmap to the original image
            alpha = 0.7
            final = alpha* u_img * (1 - colored_ht[:, :, :3]) + colored_ht[:, :, :3] * 255
            final = final.astype(np.uint8)

            # Save or display the result
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))

            # Original Image
            img_plot = axs[0].imshow(u_img)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            #axs[0].set_title(mag)

            # Heatmap
            heatmap_plot = axs[1].imshow(final, cmap='hot', norm=Normalize(vmin=0, vmax=1))  # Set 'hot' colormap here
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            #axs[1].set_title('Heatmap')

            # Add color bar to the right of the subplots
            cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.25])  # Adjust these values as needed
            cbar = fig.colorbar(heatmap_plot, cax=cbar_ax)
            default_path = path_to_save + slide + '/' + mil_model_name + '/' + encoder_name + '/'
            os.makedirs(default_path, exist_ok=True)
            fig.savefig(default_path + coords + '_' + mag, dpi=360)
            plt.show()
