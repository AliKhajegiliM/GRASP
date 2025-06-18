import json
import os
import glob
import pandas as pd
import sklearn as sk
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
import numpy as np

dataset = 'ESCA'
manifest_path = '/projects/ovcare/classification/Ali/Heram/Dataset/'+ dataset +'_dataset/scripts/'+dataset+'_manifest.csv'
path_to_save = '/projects/ovcare/classification/Ali/Heram/Dataset/'+ dataset +'_dataset/cross_validation/'
list_models = ['densenet121', 'resnet50', 'vit', 'swin','PLIP']
num_folds = 3
dataset_type = 'internal' # internal or external
stage_flag = False

def gladiator(data, data_folds, n_splits=3, train_size=0.70, random_state=256):
    group_kfold = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
    group = data.patient_id.tolist()
    group_kfold.get_n_splits(X=data, y=data.subtype, groups=group)
    count = 0
    for train_index, test_index in group_kfold.split(data, groups=group):
        count = count + 1
        #print("Train:", len(train_index), "Test:", len(test_index))
        X_train = data.iloc[train_index, :]
        #y_train = data.subtype[train_index]
        data_folds['fold-' + str(count)]['feature_path']['train'] = X_train.slide_path.tolist()
        data_folds['fold-' + str(count)]['slide_id']['train'] = X_train.slide_id.tolist()
        if stage_flag:
            data_folds['fold-' + str(count)]['label']['train'] = X_train.stage.tolist()
        else:
            data_folds['fold-' + str(count)]['label']['train'] = X_train.subtype.tolist()

        #print(X_train.patient_id, y_train)
        X_test_val = data.iloc[test_index, :]
        y_test_val = data.subtype[test_index]
        tv_group = X_test_val.patient_id.tolist()
        tv_gss = GroupShuffleSplit(n_splits=1, train_size=.50, random_state=random_state)
        tv_gss.get_n_splits(X=X_test_val, y=X_test_val.subtype, groups=tv_group)
        for test_index, val_index in tv_gss.split(X_test_val, groups=tv_group):
            print("Train:", len(train_index), "Test:", len(test_index), "Val:", len(val_index))
            X_test = X_test_val.iloc[test_index, :]
            #y_test = X_test_val.subtype.iloc[test_index]

            X_val = X_test_val.iloc[val_index, :]
            #y_val = X_test_val.subtype.iloc[val_index]
        data_folds['fold-' + str(count)]['feature_path']['val'] = X_val.slide_path.tolist()
        data_folds['fold-' + str(count)]['slide_id']['val'] = X_val.slide_id.tolist()
        if stage_flag:
            data_folds['fold-' + str(count)]['label']['val'] = X_val.stage.tolist()
        else:
            data_folds['fold-' + str(count)]['label']['val'] = X_val.subtype.tolist()
        data_folds['fold-' + str(count)]['feature_path']['test'] = X_test.slide_path.tolist()
        data_folds['fold-' + str(count)]['slide_id']['test'] = X_test.slide_id.tolist()
        if stage_flag:
            data_folds['fold-' + str(count)]['label']['test'] = X_test.stage.tolist()
        else:
            data_folds['fold-' + str(count)]['label']['test'] = X_test.subtype.tolist()

    return data_folds

for model in list_models:
    data_folds = {}
    for it in range(num_folds):
        if dataset_type == 'internal':
            data_folds['fold-'+str(it+1)] = {}
            data_folds['fold-'+str(it+1)]['slide_id'] = {}
            data_folds['fold-'+str(it+1)]['slide_id']['train'] = []
            data_folds['fold-'+str(it+1)]['slide_id']['val'] = []
            data_folds['fold-'+str(it+1)]['slide_id']['test'] = []
            data_folds['fold-'+str(it+1)]['feature_path'] = {}
            data_folds['fold-'+str(it+1)]['feature_path']['train'] = []
            data_folds['fold-'+str(it+1)]['feature_path']['val'] = []
            data_folds['fold-'+str(it+1)]['feature_path']['test'] = []
            data_folds['fold-'+str(it+1)]['label'] = {}
            data_folds['fold-'+str(it+1)]['label']['train'] = []
            data_folds['fold-'+str(it+1)]['label']['val'] = []
            data_folds['fold-'+str(it+1)]['label']['test'] = []
        elif dataset_type == 'external':
            data_folds['fold-'+str(it+1)] = {}
            data_folds['fold-'+str(it+1)]['slide_id'] = {}
            data_folds['fold-'+str(it+1)]['slide_id']['test'] = []
            data_folds['fold-'+str(it+1)]['feature_path'] = {}
            data_folds['fold-'+str(it+1)]['feature_path']['test'] = []
            data_folds['fold-'+str(it+1)]['label'] = {}
            data_folds['fold-'+str(it+1)]['label']['test'] = []
        else:
            raise NotImplementedError(f"Yo bro! {dataset_type} is neither internal nor external: go think to your bad decisions (angry emoji!)")
    
    data_manifest = pd.read_csv(manifest_path)
    model_path = []
    for item in data_manifest.slide_path.tolist():
        path_modified = item.split('/')
        path_modified[-2] = model
        data_format = path_modified[-1].split('.')[-1]
        model_path.append(os.path.join('/',*path_modified))
        #print(model_path)
    data_manifest.slide_path = model_path

    gladiator(data_manifest, data_folds, n_splits=num_folds, train_size=0.70, random_state=256)

    with open(path_to_save + model + '_data_folds_pt.json', 'w') as fp:
        json.dump(data_folds, fp)
