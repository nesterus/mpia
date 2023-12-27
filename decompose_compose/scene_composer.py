import os
import random
import math
import pandas as pd
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt

from decompose_compose.pipeline import *

from augmentation.multipart_augmentations import *
from configs.augmentation_config import aug_list
from utils.paste import *


def _create_empty_img(self, df_stat, part_id, input_type):
    ''' For debugging '''
    h = df_stat['part_height'].values[part_id]
    w = df_stat['part_width'].values[part_id]
    if input_type == 'mask':
        img = np.ones((h, w)).astype(int)
    elif input_type == 'img':
        img = np.ones((h, w, 3)).astype(int)
    return img


def _insert_part_mask(transformed_parts, obj_id, part_type, part, img, min_y, min_x, input_type):
    ''' Embeds part's image or mask into the shape of the object'''
    transformation = 'sampled_parts'

    # y0, y1, x0, x1 are the coordinates of the part's boundary
    y0 = transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['y0_coord'] - min_y
    y1 = y0 + transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['part_height']
    x0 = transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['x0_coord'] - min_x
    x1 = x0 + transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['part_width']
    
    try:
        # sampled object
        if input_type == 'mask':
            mask_target = np.zeros((img.shape[0], img.shape[1])).astype(int)
            mask_part = transformed_parts[obj_id][part_type][part][transformation]['mask'] #[part_id]        

        elif input_type == 'img':
            ch_dim = transformed_parts[obj_id][part_type][part][transformation]['img'].shape[2]
            mask_target = np.zeros((img.shape[0], img.shape[1], ch_dim)).astype(int)
            mask_part = transformed_parts[obj_id][part_type][part][transformation]['img']
            
        mask_target[int(y0):int(y1), int(x0):int(x1)] = mask_part
        return mask_target
    except:
        # composed object
        transformation = 'transformed'
        
        if input_type == 'mask':
            mask_target = np.zeros((img.shape[0], img.shape[1])).astype(int)
            mask_part = transformed_parts[obj_id][part_type][part][transformation]['mask'] #[part_id]        

        elif input_type == 'img':
            ch_dim = transformed_parts[obj_id][part_type][part][transformation]['img'].shape[2]
            mask_target = np.zeros((img.shape[0], img.shape[1], ch_dim)).astype(int)
            mask_part = transformed_parts[obj_id][part_type][part][transformation]['img']

        y0 = min_y
        y1 = min_y + mask_part.shape[0]
        x0 = min_x
        x1 = min_x + mask_part.shape[1]

        try:
            mask_part = mask_part[:img.shape[0], :img.shape[1], :]
        except:
            mask_part = mask_part[:img.shape[0], :img.shape[1]]
            
        try:
            mask_target[int(y0):int(y1), int(x0):int(x1)] = mask_part
            return mask_target
        except:
            print('Skipped part transformation')
            img = transformed_parts[obj_id][part_type][part][transformation][input_type]
            return img

def _obj_hw(transformed_parts, obj_id):
    ''' 
    Computes height and width of the object
    and coordinates (y, x) of the top left pixel of the object
    '''
    h = list() #list height heights of each part
    w = list() #list with heights of each part
    y0 = list() #list the left top y pixel for each part
    x0 = list() #list the left top x pixel for each part
    for part_type in transformed_parts[obj_id].keys():
        for part in range(len(transformed_parts[obj_id][part_type])):           
            h.append(transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['part_height'])
            w.append(transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['part_width'])
            x0.append(transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['x0_coord'])
            y0.append(transformed_parts[obj_id][part_type][part]['target_coords']['real_plant']['y0_coord'])
    
    img_h = max(np.add(y0, h)) - min(y0)
    img_w = max(np.add(x0, w)) - min(x0)
    return int(img_h), int(img_w), int(min(y0)), int(min(x0))


def _compose_object(transformed_parts, obj_id):
    ''' 
    Create image with all object's parts and an empty background.
    Also returns a list of masks with the shape (num_parts, height, width)
    where each part's mask is presented as an individual layer.
    '''
    mask_new = list()
    h, w, min_y, min_x = _obj_hw(transformed_parts, obj_id)
    
    try:
        ch_dim = transformed_parts[0][list(transformed_parts[0].keys())[0]][0]['sampled_parts']['img'].shape[2]
    except:
        ch_dim = 3
    
    img_new = np.ones((h, w, ch_dim)).astype(int)
    for part_type in transformed_parts[obj_id].keys():
        for part in range(len(transformed_parts[obj_id][part_type])):
            img_target = _insert_part_mask(transformed_parts, obj_id, part_type, part, img_new, min_y, min_x, 'img')
            mask_target = _insert_part_mask(transformed_parts, obj_id, part_type, part, img_new, min_y, min_x, 'mask')
            
            if (mask_target.shape[:2] != img_new.shape[:2]) or (img_target.shape[:2] != img_new.shape[:2]):
                continue
            
            img_new = np.where(np.dstack([mask_target>0]*ch_dim), img_target * np.dstack([mask_target]*ch_dim), img_new)

            mask_new.append(mask_target * part_type)
    return img_new, mask_new, min_y, min_x


def _split_canvas_horizontaly(h_background, w_background, num_objs):
    y0_center = list()
    x0_center = list()
    cell_width = w_background // num_objs
    cell_height = h_background
    for ind in range(num_objs):
        y0_center.append(h_background // 2)
        x0_center.append(ind*cell_width + cell_width // 2)
    return y0_center, x0_center, cell_height, cell_width


def _split_canvas2d(h_background, w_background, num_objs):
    y0_center_list = list()
    x0_center_list = list()
    
    num_cells_w = int(math.sqrt(num_objs - 1) + 1)
    num_cells_h = math.ceil(num_objs / num_cells_w)
    
    cell_width = w_background // num_cells_w
    cell_height = h_background // num_cells_h
    
    for ind in range(num_objs):
        row_idx = ind // num_cells_w
        col_idx = ind % num_cells_w
                
        y0_center_list.append((row_idx * cell_height) + (cell_height // 2))
        x0_center_list.append((col_idx * cell_width) + (cell_width // 2))
        
    return y0_center_list, x0_center_list, cell_height, cell_width


def _split_canvas_random(h_background, w_background, num_objs):
    y0_center = list()
    x0_center = list()
    cell_width = w_background // int(math.sqrt(num_objs))
    cell_height = h_background // int(math.sqrt(num_objs))
    
    assert (cell_width > 5) and (cell_height > 5)
    
    for ind in range(num_objs):
        h = random.randint((cell_height//2), (h_background - (cell_height//2)))
        w = random.randint((cell_width//2), (w_background - (cell_width//2)))
        y0_center.append(h)
        x0_center.append(w)
        
    return y0_center, x0_center, cell_height, cell_width


def _overlap_ratio(h1, w1, h2, w2, h, w):
    x = (abs(w1 - w2) - w) / w
    y = (abs(h1 - h2) - h) / h
    return max(x, y)


def _split_canvas_random_no_overlap(h_background, w_background, num_objs, overlap_thres=0.2, attempts_per_object=100):
    y0_center = list()
    x0_center = list()
    cell_width = w_background // int(math.sqrt(num_objs))
    cell_height = h_background // int(math.sqrt(num_objs))
    
    assert (cell_width > 5) and (cell_height > 5)
    
    for ind in range(num_objs):
        for repetition in range(attempts_per_object):
            h = random.randint((cell_height//2), (h_background - (cell_height//2)))
            w = random.randint((cell_width//2), (w_background - (cell_width//2)))
            
            is_good_place = True
            for i in range(len(y0_center)):
                if _overlap_ratio(h, w, y0_center[i], x0_center[i], cell_height, cell_width) < overlap_thres:
                    is_good_place = False
                    break
                    
            if is_good_place:
                y0_center.append(h)
                x0_center.append(w)
    
    non_overlap_objects = len(y0_center)
    overlapping_objects = num_objs - non_overlap_objects
    if overlapping_objects > 0:
        print('{} objects will overlap more than {}%'.format(overlapping_objects, int(overlap_thres*100)))
        y0_center_aux, x0_center_aux, _, _ = _split_canvas_random(h_background, w_background, overlapping_objects)
        y0_center.extend(y0_center_aux)
        x0_center.extend(x0_center_aux)
        
    return y0_center, x0_center, cell_height, cell_width


def _convert_obj_list2dict(obj_img, obj_mask):
    obj_dict = {}
    obj_dict['img'] = obj_img.astype(np.uint8)
    obj_dict['mask'] = np.zeros((obj_img.shape[0], obj_img.shape[1])).astype(np.uint8)
    obj_dict['parts_list'] = obj_mask
    return obj_dict


def _insert_obj2background(obj_dict, background, y0_center, x0_center, obj_id, blending_mode):
    full_mask = np.array(obj_dict['parts_list']).sum(axis=0)
    ch_dim = background.shape[2]

    y0 = y0_center[obj_id] - obj_dict['img'].shape[0] // 2 #top 
    y1 = y0 + obj_dict['img'].shape[0]
    x0 = x0_center[obj_id] - obj_dict['img'].shape[1] // 2 #left
    x1 = x0 + obj_dict['img'].shape[1]
    
    mask_ext = np.zeros((background.shape[0], background.shape[1])).astype(np.uint8)
    mask_ext[y0:y1, x0:x1] = full_mask 
    img_ext = np.zeros(background.shape).astype(np.uint8)
    img_ext[y0:y1, x0:x1] = obj_dict['img'] 
    mask_part_list = list()
        
    for mask_part in obj_dict['parts_list']:

        mask_ext_part = np.zeros((background.shape[0], background.shape[1])).astype(np.uint8)
        mask_ext_part[y0:y1, x0:x1] = mask_part
        mask_part_list.append(mask_ext_part)
    
    if blending_mode == 'poisson':
        source = img_ext
        source_mask = ((mask_ext > 0) * 255).astype(np.uint8)
        
        try:
            background = paste_object_poisson(background.astype(np.uint8), source.astype(np.uint8), source_mask.astype(np.uint8), x=0, y=0, backend="taichi-gpu")
        except:
            background = paste_object_poisson(background.astype(np.uint8), source.astype(np.uint8), source_mask.astype(np.uint8), x=0, y=0, backend="numpy")

    if blending_mode == 'base':
        img_new = np.where(np.dstack([mask_ext>0]*ch_dim), img_ext * np.dstack([mask_ext>0]*ch_dim), background)
    elif blending_mode == 'poisson':
        img_new = background
    return img_new, mask_part_list


class SceneComposer:
    def __init__(
        self, 
        project_config=None,
        project_config_path=None
    ): 
        if project_config:
            self.project_config = project_config
        else:
            with open(project_config_path, 'r') as f:
                self.project_config = json.loads(f.read())
        
    def compose_img(self, objects, background):
        ''' Composes a new image based on a set of input objects and new background '''
        Augmentor = DatasetAugmentor(project_config=self.project_config)

        num_objs = len(objects)
        h_background = background.shape[0]
        w_background = background.shape[1]

        if self.project_config['object_placement'] == 'uniform_2d':
            y0_center, x0_center, cell_height, cell_width = _split_canvas2d(h_background, \
                                                                                      w_background, \
                                                                                      num_objs)
        elif self.project_config['object_placement'] == 'uniform_1d':
            y0_center, x0_center, cell_height, cell_width = _split_canvas_horizontaly(h_background, \
                                                                                      w_background, \
                                                                                      num_objs)
        elif self.project_config['object_placement'] == 'random':
            y0_center, x0_center, cell_height, cell_width = _split_canvas_random(h_background, \
                                                                                      w_background, \
                                                                                      num_objs)
        elif self.project_config['object_placement'] == 'no_overlap':
            y0_center, x0_center, cell_height, cell_width = _split_canvas_random_no_overlap(h_background, \
                                                                                      w_background, \
                                                                                      num_objs)
        else:
            raise ValueError('placement mode {} id not supproted'.format(self.project_config['object_placement']))

        max_size = min(cell_height, cell_width)
        aug_lms = {'pipeline': A.Compose([A.LongestMaxSize(max_size)])}

        mask_new = list()
        for obj_id in range(len(objects)):
            obj_img, obj_mask, min_y, min_x = _compose_object(objects, obj_id)

            obj_dict = _convert_obj_list2dict(obj_img, obj_mask)

            ###

            obj_h = obj_dict['img'].shape[0] 
            obj_w = obj_dict['img'].shape[1] 
            cell_h = cell_height
            cell_w = cell_width

            min_h_ratio = cell_h / obj_h
            min_w_ratio = cell_w / obj_w
            min_ratio = min(min_h_ratio, min_w_ratio)

            if min_ratio >= 1:
                ratio = random.uniform(1.0, min_ratio)
            else:
                ratio = min_ratio

            ###

            h = int(obj_dict['img'].shape[0] * ratio) - 1
            w = int(obj_dict['img'].shape[1] * ratio) - 1

            aug_r = {'pipeline': A.Compose([A.Resize(h, w, interpolation=4)])}
            aug_obj_dict = Augmentor.augment_object(obj_dict, aug_r)
            
            try:
                img_new, mask_part_list = _insert_obj2background(aug_obj_dict, background, y0_center, x0_center, obj_id, self.project_config['blending_mode'])
            except:
                if self.project_config['verbose'] >= 1:
                    print('Skipping too small object')

            background = img_new
            mask_new.append(mask_part_list)

        return img_new, mask_new 
