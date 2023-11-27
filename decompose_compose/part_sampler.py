import json
import numpy as np
from PIL import Image
import cv2
import random
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from metadata.metadata import object_type
from metadata.polygon_masks import rleToMask
from utils.individual_obj_mask import crop_obj_bb


def part_sampler(dataset_dir, class_type, visualize=False):
    part_stat_path = dataset_dir + 'part_statistics.csv'
    df_part = read_csv(part_stat_path)
    
    object_type_switched = {y: x for x, y in object_type.items()}
    df_part['class_type'] = df_part['class_type'].map(object_type_switched)
    
    df_filtered = df_part[df_part['class_type'] == class_type]
    
    parts_ids = df_filtered['id_names'].to_list() 
    
    sampler_dict = {}
    for part_id in parts_ids:
        sampler_dict[part_id] = {}
        part_stat = df_filtered[df_filtered['id_names'] == part_id]
        sampler_dict[part_id]['part_stat'] = part_stat
        img_format = part_stat['id_names'].values[0].split('.')[1]
        file = str(part_stat['img_id'].values[0]) + '.{}_mpta.json'.format(img_format)
        file_path = open(dataset_dir + 'ann/' + file)
        file_stat = json.load(file_path)

        i = int(part_id.split('_')[-1])
        rle_mask = file_stat['mask'][i]
        mask = rleToMask(rle_mask, file_stat['width'], file_stat['height'])
        mask = (mask > 0).astype(np.uint8)
        mask_crop = crop_obj_bb(mask, mask)
        sampler_dict[part_id]['mask'] = mask_crop
        
        if visualize:
            plt.imshow(mask_crop)
            plt.show()

        img = cv2.imread(dataset_dir + 'img/' + str(part_stat['img_id'].values[0]) + '.' + img_format)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = crop_obj_bb(img, mask)
        sampler_dict[part_id]['img'] = img
        
        if visualize:
            plt.imshow(img)
            plt.show()

    return sampler_dict 


def random_part_sampler(dataset_list, class_type, min_height=0, min_width=0, visualize=False):
    # select dataset
    dataset_dir = random.choice(dataset_list)
    
    # select sample
    part_stat_path = dataset_dir + 'part_statistics.csv'
    df_part = read_csv(part_stat_path)
    
    object_type_switched = {y: x for x, y in object_type.items()}
    df_part['class_type'] = df_part['class_type'].map(object_type_switched)
    
    df_filtered = df_part[df_part['class_type'] == class_type]
    
    # filtered minimum part size
    if min_height>0 or min_width>0:
        df_filtered = df_filtered[df_filtered['part_height'] > min_height]
        df_filtered = df_filtered[df_filtered['part_width'] > min_width]
        
        if len(df_filtered.index) == 0:
            max_height = df_part[df_part['part_height']==df_part['part_height'].max()]['part_height'].values[0]
            max_width = df_part[df_part['part_width']==df_part['part_width'].max()]['part_height'].values[0]
            print('The min_height or min_width is larger than maximum height and width for this part type. \n \
            The maximum height and width are {}, {}'.format(max_height, max_width))
            
            return None
    
    parts_ids = df_filtered['id_names'].to_list() 
    
    sampler_dict = {} #part_id
    
    part_id = random.choice(parts_ids)
    
    part_stat = df_filtered[df_filtered['id_names'] == part_id]
    sampler_dict['part_stat'] = part_stat
    img_format = part_stat['id_names'].values[0].split('.')[1]
    file = str(part_stat['img_id'].values[0]) + '.{}_mpta.json'.format(img_format)
    file_path = open(dataset_dir + 'ann/' + file)
    file_stat = json.load(file_path)

    i = int(part_id.split('_')[-1])
    rle_mask = file_stat['mask'][i]
    mask = rleToMask(rle_mask, file_stat['width'], file_stat['height'])
    mask = (mask > 0).astype(np.uint8)
    mask_crop = crop_obj_bb(mask, mask)
    sampler_dict['mask'] = mask_crop

    if visualize:
        plt.imshow(mask_crop)
        plt.show()

    img = cv2.imread(dataset_dir + 'img/' + str(part_stat['img_id'].values[0]) + '.' + img_format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = crop_obj_bb(img, mask)
    sampler_dict['img'] = img

    if visualize:
        plt.imshow(img)
        plt.show()

    return sampler_dict 


class PartSampler:
    def __init__(self, project_config, min_height=0, min_width=0):
        self.project_config=project_config
        self.min_height = min_height
        self.min_width = min_width
    
    def sample_all_dataset(self, dataset_name, class_type):
        dataset_dir = './datasets/{}/{}/'.format(dataset_name, dataset_name)
        return part_sampler(dataset_dir, class_type=class_type, visualize=False)
    
    def sample(self, class_type):
        dataset_names = self.project_config['dataset_names']
        dataset_list = []
        for ds in dataset_names:
            dataset_list += ['./datasets/{}/{}/'.format(ds, ds)]
        return random_part_sampler(dataset_list=dataset_list, class_type=class_type, min_height=self.min_height, min_width=self.min_width, visualize=False)
