from pathlib import Path
import pandas as pd
import numpy as np
import os

import metadata
from decompose_compose.feature_sampling import *
from decompose_compose.part_sampler import *
from decompose_compose.dataset_statistics import *
from decompose_compose.object_sampler import *
from decompose_compose.scene_composer import *
from background_generation.background_generation import *
from utils.report_generator import create_stat_report
from utils.utils import *
from augmentation.spatial_augmentations import *
from augmentation.multipart_augmentations import *
from augmentation.object_schema_augmentation import *


class Pipeline:
    def __init__(
        self, 
        project_config=None,
        project_config_path=None,
        feature_sampler_limitations=None,
        verbose=0
    ):        
        if project_config:
            self.project_config = project_config
        else:
            with open(project_config_path, 'r') as f:
                self.project_config = json.loads(f.read())
                
        self.project_config['verbose'] = verbose
            
        self.background_sampler = BackgroundSampler(self.project_config)
        self.feature_sampler = FeatureSampler(
            project_config = self.project_config,
            limitations = feature_sampler_limitations
        )
        self.part_sampler = PartSampler(self.project_config, min_height=20, min_width=20)
        self.object_sampler = ObjectSampler(project_config=self.project_config)
        self.augmentor = DatasetAugmentor(project_config=self.project_config)
        self.scene_composer = SceneComposer(project_config=self.project_config)
        
        if self.project_config['verbose'] >= 1:
            print(self.project_config)
        
    def prepare(self):
        '''
        Calculates all required dataset statistics, generates backgrounds, makes all verifications.
        May take several minutes.
        '''
        for dataset_name in self.project_config['dataset_names']:
            dataset_dir = './datasets/{}/{}/'.format(dataset_name, dataset_name)
            file_list = os.listdir(dataset_dir)
            
            if ('part_statistics.csv' in file_list) and ('object_statistics.csv' in file_list):
                continue
                
            dataset_statistics(dataset_dir) # create files with parts and objects statistics
            add_dist_stat(dataset_dir) # add distance statistics for each part
            
        self._pregenerate_backgrounds()
        
    def generate_report(self):
        for dataset_name in self.project_config['dataset_names']:
            create_stat_report(dataset_name) # create pdf files with statistics for dataset
        
    def run(self, num_objects=4, return_at_least_one=True, max_reties=10):
        ''' Makes single augmented image '''
        mode = self.project_config['composition_mode']
        background_image = self.background_sampler.sample_from_folder()
        if not num_objects:
            num_objects = self.feature_sampler.sample_objects_num()
        
        if mode == 'creation':
            objects_list = list()
            for obj_idx in range(num_objects):
                object_schema = self.feature_sampler.make_object_schema()
                parts_data = self._sample_parts_to_schema(object_schema)

                for part_type_num in parts_data:
                    for part_idx, part in enumerate(parts_data[part_type_num]):
                        parts_data[part_type_num][part_idx] = self.augmentor.augment_part(part)

                parts_data = self._transform_parts_to_target(parts_data)
                objects_list.append(parts_data)
        
        elif mode == 'modification':
            objects_list = list()
            for obj_idx in range(num_objects):
                parts_data, real_plant = self.object_sampler.sample_object(self.feature_sampler.part_df)
                object_schema = self.feature_sampler.sample_part_features_copy(real_plant.copy(deep=True))
                
                for part_type in object_schema:
                    for part_idx, part in enumerate(object_schema[part_type]):
                        parts_data[part_type][part_idx]['target_coords'] = object_schema[part_type][part_idx]['target_coords']
                parts_data = augment_object_schema(parts_data)
                parts_data = self._transform_parts_to_target(parts_data)
                
                objects_list.append(parts_data)
        else:
            raise ValueError('mode {} id not supproted'.format(mode))
                
        generated_scene, mask_part_list = self.scene_composer.compose_img(objects_list, background_image)
        
        if len(mask_part_list[0]) < 1 and return_at_least_one and max_reties:
            if self.project_config['verbose']:
                print('Bad object placement. Retrying.')
            return self.run(num_objects=num_objects, return_at_least_one=True, max_reties=max_reties-1)
        
        return generated_scene, mask_part_list
        
    def _transform_parts_to_target(self, object_schema):
        
        for part_type in object_schema:
            for part_idx, part_sample_target in enumerate(object_schema[part_type]):
                
                sample = object_schema[part_type][part_idx]['sampled_parts']
                target = object_schema[part_type][part_idx]['target_coords']
                
                transform_params = {
                    'img': sample['img'].astype(np.uint8), 
                    'mask': sample['mask'].astype(np.uint8), 
                    'init_angle': int(sample['angle']), 
                    # keypoints=sample['connection_points'], 
                    'keypoints': [],
                    'target_angle': int(target['angle']), 
                    'target_h': int(target['h']), 
                    'target_w': int(target['w'])
                }

                transformed_res = mild_spatial_transform(**transform_params)
                object_schema[part_type][part_idx]['transformed'] = {
                    'img': transformed_res['img'],
                    'mask': transformed_res['mask']
                }
                
        return object_schema
        
    def _sample_parts_to_schema(self, object_schema):
        ''' 
        Samples parts for each element in provided schema.
        If elements in schema already has sampled_parts, returns as-is.
        '''
        if 'sampled_parts' in object_schema[list(object_schema.keys())[0]][0]:
            return object_schema
        for part_type in object_schema:
            for part_idx, part_sample_target in enumerate(object_schema[part_type]):
                class_type_name = reverse_dict(metadata.metadata.object_type)[part_type]
                part_sample_data = self.part_sampler.sample(class_type_name)
                object_schema[part_type][part_idx]['sampled_parts'] = {
                    'img': part_sample_data['img'],
                    'mask': part_sample_data['mask'],
                    'h': int(part_sample_data['part_stat']['main_diag_height']), # main_diag_height part_height
                    'w': int(part_sample_data['part_stat']['main_diag_width']), # main_diag_width part_width
                    'x': int(part_sample_data['part_stat']['x_coord_object']),
                    'y': int(part_sample_data['part_stat']['y_coord_object']),
                    'angle': int(part_sample_data['part_stat']['alpha_horizons']),
                    # 'connection_points': part_sample_data['part_stat']['connection_points'],
                    'stats': part_sample_data['part_stat'],
                    'real_plant': part_sample_data
                }
                
        return object_schema
        
    def _pregenerate_backgrounds(self):
        ''' 
        Iterates over images in datasets. For each image, removes all objects using their masks.
        Describes the resulting background. And generated background images with the same description.
        Saves them to a folder.
        '''
        Path(self.project_config['background_generation']['storage_path']).mkdir(parents=True, exist_ok=True)
        
        existing_files = os.listdir(self.project_config['background_generation']['storage_path'])
        existing_files = [f for f in existing_files if ('.jpg' in f)]
        
        if (len(existing_files) == 0) and (self.project_config['device'] == 'cpu'):
            raise FileNotFoundError('Not background images found in {}. Can not generate them automatically with device type "CPU"'.format(self.project_config['background_generation']['storage_path']))
        
        if not self.project_config['force_repreparation']:
            if len(existing_files) > 0:
                if self.project_config['verbose'] >= 1:
                    print('Using existing backgrounds from {}'.format(self.project_config['background_generation']['storage_path']))
                return
        
        for dataset_name in self.project_config['dataset_names']:
            dataset_dir = './datasets/{}/{}/'.format(dataset_name, dataset_name)
            file_list = os.listdir(dataset_dir + 'ann/')
            
            samples_per_dataset = self.project_config['background_generation']['samples_per_dataset']
            
            file_list = [file for file in os.listdir(dataset_dir + 'ann/') if file[-9:] == 'mpta.json']
            for file in file_list:
                if samples_per_dataset == 0:
                    break
                    
                try:
                    file_path = open(dataset_dir + 'ann/' + file)
                    file_stat = json.load(file_path)

                    full_mask = np.zeros((file_stat['height'], file_stat['width']))
                    for i in range(len(file_stat['mask'])):
                        rle_mask = file_stat['mask'][i]
                        mask = rleToMask(rle_mask, file_stat['width'], file_stat['height'])
                        full_mask += np.array(mask, dtype=np.uint8)

                    full_mask = full_mask > 0
                    img = cv2.imread(dataset_dir + 'img/' + file.split('.')[0] + '.jpg')
                        
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_masked = np.copy(img)
                    img_masked[full_mask] = 0

                    upscaled_background = self.background_sampler.generate_online(img_masked, save=True)
                    samples_per_dataset -= 1
                except cv2.error:
                    pass
