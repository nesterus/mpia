from runpy import run_path
import albumentations as A
from matplotlib import pyplot as plt
import numpy as np
import random
import os
import copy
import json


class DatasetAugmentor:
    def __init__(self, project_config=None, project_config_path=None,):
        if project_config:
            self.project_config = project_config
        else:
            with open(project_config_path, 'r') as f:
                self.project_config = json.loads(f.read())
            
        self.augmentation_config = run_path(self.project_config['augmentations_config_path'])

    def augment_objects(self, objects_list):
        augmented_objects_list = list()
        
        for obj_idx, obj_dict in enumerate(objects_list):
            for aug in self.augmentation_config['aug_list']:
                if self._is_aug_applicable(obj_dict, aug, level='object'):
                    augmented_object = self.augment_object(obj_dict, aug)
                    augmented_objects_list.append(augmented_object)
                    
                    if aug['terminate']:
                        break
                    
        return augmented_objects_list
    
    def augment_background(self, img):
        for aug in self.augmentation_config['aug_list']:
            if self._is_aug_applicable(obj=None, aug=aug, level='background'):
                transform = aug['pipeline']
                transformed = transform(image=img)  
                return transformed['image']
        return img
    
    def _is_aug_applicable(self, obj, aug, level='object_part'):
        if level not in aug['target']:
            return False
        
        if level == 'background':
            return True
        else:
            obj_class = obj['obj_class_name']
            is_applicable_class = False
            if isinstance(aug['apply_to_classes'], str) and aug['apply_to_classes'] == 'all':
                is_applicable_class = True
            else:
                is_applicable_class = any([obj_class==applicable_class for applicable_class in aug['apply_to_classes']])
            exclude_class = any([obj_class==applicable_class for applicable_class in aug['exclude_classes']])
            is_applicable_class = is_applicable_class and not exclude_class
            
        if level == 'object':
            return is_applicable_class # TODO: add tags
        elif level == 'object_part':
            part_type = obj['obj_part_name']
            is_applicable_part_type = False
            if isinstance(aug['apply_to_object_types'], str) and aug['apply_to_object_types'] == 'all':
                is_applicable_part_type = True
            else:
                is_applicable_part_type = any([part_type==applicable_part_type for applicable_part_type in aug['apply_to_object_types']])
            exclude_part_type = any([part_type==applicable_part_type for applicable_part_type in aug['exclude_object_types']])
            is_applicable_part_type = is_applicable_part_type and not exclude_part_type
            
            return is_applicable_class and is_applicable_part_type  # TODO: add tags
    
    def augment_object(self, obj_dict, aug):
        '''parts simultaniously '''
        new_obj_dict = copy.deepcopy(obj_dict)
        all_masks_list = [obj_dict['mask']]
        all_masks_list.extend(obj_dict['parts_list'])
        all_masks = np.stack(all_masks_list, axis=2)
        
        transform = aug['pipeline']
        transformed = transform(
            image=obj_dict['img'],
            mask=all_masks
        )
                
        new_obj_dict['img'] = transformed['image']
        augmented_all_masks = transformed['mask']
        augmented_all_masks_list = [augmented_all_masks[:, :, i] for i in range(augmented_all_masks.shape[-1])]
        new_obj_dict['mask'] = augmented_all_masks_list[0]
        new_obj_dict['parts_list'] = augmented_all_masks_list[1:]
        
        return new_obj_dict

    def augment_part(self, part_obj):  
        if 'sampled_parts' not in part_obj:
            return part_obj
        
        obj_dict = {
            'obj_class_name': part_obj['sampled_parts']['stats']['class'].values[0],
            'obj_part_name': part_obj['sampled_parts']['stats']['class_type'].values[0],
        }
        
        for aug in self.augmentation_config['aug_list']:
            if self._is_aug_applicable(obj_dict, aug, level='object_part'):
                transform = aug['pipeline']
                transformed = transform(
                    image=part_obj['sampled_parts']['img'],
                    mask=part_obj['sampled_parts']['mask']
                )
                
                part_obj['sampled_parts']['img'] = transformed['image']
                part_obj['sampled_parts']['mask'] = transformed['mask']
                
                if aug['terminate']:
                    break
                
        return part_obj
