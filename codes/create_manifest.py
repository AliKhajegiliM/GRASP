import os
import glob
import pandas as pd

if __name__ == '__main__':
    data_locations = ['/projects/ovcare/classification/TCGA_features/Diagnostics/ESCA/vit/']
    data_path = []
    data = {'origin': [],
       'patient_id': [],
       'slide_id': [],
        'subtype': [],
       'slide_path': []}
    for data_location in data_locations:
        path_wildcard = os.path.join(data_location,  '**','*.' + 'h5')
        data_path.extend(glob.glob(path_wildcard, recursive=True))
    for item in data_path:
        origin = 'other'
        ent = item.split('/')
        patient_id = ent[-1].split('.')[0][:-4]
        slide_id = ent[-1][:-3]
        subtype = ent[-3]
        slide_path = item

        data['origin'].append(origin)
        data['patient_id'].append(patient_id)
        data['slide_id'].append(slide_id)
        data['subtype'].append(subtype)
        data['slide_path'].append(slide_path)
    
    data_frame = pd.DataFrame(data)
    data_frame.to_csv("/projects/ovcare/classification/Ali/Heram/Dataset/ESCA_dataset/cross_validation/ESCA_manifest.csv", index=False)