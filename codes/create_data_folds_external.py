import json
main_path = '/projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/features/test_sets/private_all/zero-shot/OV/'
path_to_manifest = '/projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/scripts/manifests/Private.csv'

import os
import glob
import pandas as pd

df = pd.read_csv(path_to_manifest)

for dataset in ['OV']:
    for model in ["UNI", "prov-GigaPath", "H-optimus-0", "Virchow2", "Virchow1", "Phikon-v2", "Kaiko_B8"]:
        data_folds = {}
        for direction in ['fold-1', 'fold-2', 'fold-3']:
            
            num_folds = 3
            #for it in range(num_folds):
            data_folds[direction] = {}
            data_folds[direction]['slide_id'] = {}
            data_folds[direction]['slide_id']['test'] = []
            data_folds[direction]['feature_path'] = {}
            data_folds[direction]['feature_path']['test'] = []
            data_folds[direction]['label'] = {}
            data_folds[direction]['label']['test'] = []
            patch_path = []
            for mother_path in [main_path + '/' + model + '/']:
                path_wildcard = os.path.join(mother_path, '**', '*.h5')
                patch_path.extend(glob.glob(path_wildcard, recursive=True))

            print(len(patch_path))
            key = 'test'
            for item in patch_path:
                slide = item.split('/')[-1].split('.')[0]
                if not slide in data_folds[direction]['slide_id'][key]:
                    print(slide)
                    subtype = df[df['slide_id']==slide].subtype.item()
                    if subtype == 'Other' or subtype == 'other':
                        subtype='other'
                    data_folds[direction]['slide_id'][key].append(slide)
                    path_to_slide = item
                    data_folds[direction]['feature_path'][key].append(path_to_slide)
                    data_folds[direction]['label'][key].append(subtype)
                    # if subtype == 'Other' or subtype == 'other':
                    #     pass
                    # else:
                    #     data_folds[direction]['slide_id'][key].append(slide)
                    #     path_to_slide = item
                    #     data_folds[direction]['feature_path'][key].append(path_to_slide)
                    #     data_folds[direction]['label'][key].append(subtype)
        path_to_save = '/projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/cross_validation/mil_folds_zero-shot/fm' + '/private_all/' 
        os.makedirs(path_to_save, exist_ok=True)
        with open(path_to_save + model +'_' +dataset+ '_data_folds_h5.json', 'w') as fp:
            json.dump(data_folds, fp)
