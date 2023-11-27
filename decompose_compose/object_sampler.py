import os
import json
import numpy as np
from PIL import Image
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt

from metadata.metadata import object_type
from metadata.polygon_masks import rleToMask
from utils.individual_obj_mask import crop_obj_bb


class ObjectSampler:
    def __init__(self, project_config=None, project_config_path=None):

        if project_config is not None:
            self.project_config=project_config
        else:
            with open(project_config_path, 'r') as f:
                self.project_config = json.loads(f.read())
        
    def sample_object(self, df):
        real_random_obj_name = random.choice(df['obj_id'].unique())
        real_plant = df[df['obj_id'] == real_random_obj_name]
        
        dataset_dir = './datasets/{}/{}/'.format(real_plant['dataset'].values[0], real_plant['dataset'].values[0])
        
        img_format = real_plant['id_names'].values[0].split('.')[1]
        file = str(real_plant['img_id'].values[0]) + '.{}_mpta.json'.format(img_format)
        file_path = open(dataset_dir + 'ann/' + file)
        file_stat = json.load(file_path)

        parts_list = list()
        parts_ids = real_plant['id_names'].to_list() 
        for parts_id in parts_ids:
            part_dict = {}
            i = int(parts_id.split('_')[-1]) # part id
            rle_mask = file_stat['mask'][i]
            mask = rleToMask(rle_mask, file_stat['width'], file_stat['height'])
            mask = (mask > 0).astype(np.uint8)
            mask_crop = crop_obj_bb(mask, mask)
            part_dict['mask'] = mask_crop

            img = cv2.imread(dataset_dir + 'img/' + str(real_plant['img_id'].values[0]) + '.' + img_format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = crop_obj_bb(img, mask)
            part_dict['img'] = img
            
            part_dict['stats'] = real_plant[real_plant['id_names'] == parts_id]

            parts_list.append(part_dict)
            
        parts_data = self._parts_list2parts_data(parts_list)
            
        return parts_data, real_plant.copy(deep=True)
    
    def _parts_list2parts_data(self, parts_list):
        parts_data = {}
        
        part_types_list = list()
        for part in parts_list:
            part_types_list.append(part['stats']['class_type'].values[0])
        part_types_list = list(set(part_types_list))
        
        for part_type in part_types_list:
            parts_with_type_list = list()
            for part in parts_list:
                if part['stats']['class_type'].values[0] == part_type:
                    parts_with_type_list.append({'sampled_parts': {
                        'img': part['img'],
                        'mask': part['mask'],
                        'h': int(part['stats']['main_diag_height']), # main_diag_height part_height
                        'w': int(part['stats']['main_diag_width']), # main_diag_width part_width
                        'x': int(part['stats']['x_coord_object']),
                        'y': int(part['stats']['y_coord_object']),
                        'angle': int(part['stats']['alpha_horizons']),
                        # 'connection_points': part['stats']['connection_points'],
                        'stats': part['stats']
                    }})
                    
            parts_data[part_type] = parts_with_type_list
        
        return parts_data
