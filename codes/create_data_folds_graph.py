import os
import glob
import pandas as pd
import json

path_to_pt_folds = '/projects/ovcare/classification/Ali/Heram/Dataset/ESCA_dataset/cross_validation/'
pt_folds_list = []
path_wildcard = os.path.join(path_to_pt_folds,  '**','*pt.' + 'json')
pt_folds_list.extend(glob.glob(path_wildcard, recursive=True))
print(pt_folds_list)
new_mother_path = '/projects/ovcare/classification/Ali/Heram/Dataset/ESCA_dataset/graphs'
path_to_save = '/projects/ovcare/classification/Ali/Heram/Dataset/ESCA_dataset/cross_validation/'

for jsonfile in pt_folds_list:
    model = jsonfile.split('/')[-1].split('_')[0]
    f = open(jsonfile)
    data_dict = json.load(f)
    for fold in data_dict.keys():
        for set_ in data_dict[fold]['feature_path'].keys():
            modified_paths = []
            for item in data_dict[fold]['feature_path'][set_]:
                slide = item.split('/')[-1].split('.')[0]
                print(slide)
                modified_paths.append(new_mother_path + '/' + model + '/' +  slide + '.bin')
            data_dict[fold]['feature_path'][set_] = modified_paths
    with open(path_to_save + model + '_data_folds_graph.json', 'w') as fp:
        json.dump(data_dict, fp)

#