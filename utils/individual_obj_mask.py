import json
import numpy as np
import pandas as pd
import cv2
from pandas import DataFrame
from metadata.metadata import object_classes, object_type
from metadata.polygon_masks import rleToMask

def crop_obj_bb(img, extention): # crop object along the border
    height_array, width_array = np.where(extention>0)
    crop_img = img[np.min(height_array):np.max(height_array), np.min(width_array):np.max(width_array)]
    return crop_img

# def map_ind2names(df_classes, df_class_type):
#     object_classes_switched = {y: x for x, y in object_classes.items()}
#     object_type_switched = {y: x for x, y in object_type.items()}
#     class_names = df_classes.map(object_classes_switched)
#     part_names = df_class_type.map(object_type_switched)
#     return class_names, part_names

def get_image_masks(dataset_dir, file):
    '''
    function get_image_masks returns dictionary with all objects on the image and their parts
    
    dict parameters:
        mask: object's mask in np array format 
        obj_class: list with corresponding object's class 
        parts_list: list with parts' masks in np array format for each object: parts_list[obj_id][part_id]  
        parts_type: list with parts' types (leaf, stem) in np array format for each obj: parts_list[obj_id][part_id] 
        objs_class_name: ind converted to names
        parts_type_name: ind converted to names
        img: np array with object's image
        tag: object's tag
    '''
    
    path = dataset_dir + 'ann/' + file
    file_path = open(path)
    file_stat = json.load(file_path)
    height = file_stat['height']
    width = file_stat['width']
    
    #class_names, part_names = map_ind2names(file_stat['class'], file_stat['class_type'])
    object_classes_switched = {y: x for x, y in object_classes.items()}
    object_type_switched = {y: x for x, y in object_type.items()}
    file_stat['class_name'] = pd.Series(file_stat['class']).map(object_classes_switched).tolist()
    file_stat['part_names'] = pd.Series(file_stat['class_type']).map(object_type_switched).tolist()
    
    if None in file_stat['tag_nums']:
        list_of_none = np.where(np.array(file_stat['tag_nums']) == None)[0]
        for ind in list_of_none:
            file_stat['tag_nums'][ind] = 'None'
            
    obj_dict = {}
    tag_list = np.unique(file_stat['tag_nums'])
    
    masks_list = []
    parts_list = []
    obj_class = []
    obj_class_names = []
    parts_type = []
    parts_type_names = []
    
    for obj in tag_list:
        obj_dict[obj] = {}
        
        obj_parts = []
        type_list = []
        type_list_names = []
        parts_ind = np.where(np.array(file_stat['tag_nums']) == obj)
        mask = np.zeros((height, width))
        mask_extent = np.ones((height, width))
        for ind in parts_ind[0]:
            rle_mask = file_stat['mask'][ind]
            obj_mask = rleToMask(rle_mask, width, height) 
            #obj_mask = crop_obj_bb((rleToMask(rle_mask, width, height)>0).astype(int), mask_extent)
            mask += obj_mask
            obj_parts.append((obj_mask>0).astype(int))
            type_list.append(file_stat['class_type'][ind])
            type_list_names.append(file_stat['part_names'][ind])
        
        obj_dict[obj]['obj_class'] = file_stat['class'][ind]
        obj_dict[obj]['obj_class_name'] = file_stat['class_name'][ind]
        obj_dict[obj]['mask'] = crop_obj_bb((mask>0).astype(int), (mask>0).astype(int)) # mask_extent
        obj_dict[obj]['parts_type'] = type_list
        obj_dict[obj]['parts_type_name'] = type_list_names
        # obj_dict[obj]['parts_list'] = obj_parts
        obj_dict[obj]['parts_list'] = [crop_obj_bb(obj_mask, (mask>0).astype(int)) for obj_mask in obj_parts]
        
        img = cv2.imread(dataset_dir + 'img/' + file.split('.')[0] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        obj_dict[obj]['img'] = crop_obj_bb(img, (mask>0).astype(int)) 
        obj_dict[obj]['tag'] = obj
        
    return obj_dict