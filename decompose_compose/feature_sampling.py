import pandas as pd
import numpy as np
import copy
import json
import os
import random

from utils.simplify_stats import *


class FeatureSampler:
    def __init__(self, project_config=None, project_config_path=None, limitations=None):
        if project_config is not None:
            self.project_config = project_config
        else:
            with open(project_config_path, 'r') as f:
                self.project_config = json.loads(f.read())
        
        self.load_dataset_statistics()
        self.part_df, self.simple_part_df = FeatureSampler.separate_complex_and_simple(
            orig_df=self.part_df,
            object_prefix=self.project_config['object_prefix']
        )
        self.object_df, self.simple_object_df = FeatureSampler.separate_complex_and_simple(
            orig_df=self.object_df,
            object_prefix=self.project_config['object_prefix']
        )
        
        if limitations is not None:
            self.part_df = self.apply_limitations(self.part_df, limitations_dict = limitations)
        
    def load_dataset_statistics(self):
        object_df = pd.DataFrame()
        part_df = pd.DataFrame()
        
        for dataset_name in self.project_config['dataset_names']:
            dataset_dir = './datasets/{}/{}/'.format(dataset_name, dataset_name)
            
            object_df_iter = pd.read_csv(os.path.join(dataset_dir, 'object_statistics.csv'))
            object_df_iter['dataset'] = dataset_name
            if len(object_df) > 0:
                object_df = pd.concat([object_df, object_df_iter], ignore_index=True)
            else:
                object_df = object_df_iter
            self.object_df = object_df
            
            part_df_iter = pd.read_csv(os.path.join(dataset_dir, 'part_statistics.csv'))
            part_df_iter['dataset'] = dataset_name
            if len(part_df) > 0:
                part_df = pd.concat([part_df, part_df_iter], ignore_index=True)
            else:
                part_df = part_df_iter
            self.part_df = part_df
        
    def apply_limitations(self, df, limitations_dict={}):
        df_copy = df.copy(deep=True)
        
        for lim_col in limitations_dict:
            for lim_factor in limitations_dict[lim_col]:
                if lim_factor == 'isin':
                    df_copy = df_copy[df_copy[lim_col].isin(limitations_dict[lim_col][lim_factor])]
                elif lim_factor == 'le':
                    df_copy = df_copy[df_copy[lim_col].le(limitations_dict[lim_col][lim_factor])]
                elif lim_factor == 'ge':
                    df_copy = df_copy[df_copy[lim_col].ge(limitations_dict[lim_col][lim_factor])]
                elif lim_factor == 'eq':
                    df_copy = df_copy[df_copy[lim_col].eq(limitations_dict[lim_col][lim_factor])]
                elif lim_factor == 'ne':
                    df_copy = df_copy[df_copy[lim_col].ne(limitations_dict[lim_col][lim_factor])]
                else:
                    print('Ignoring unsupported limitation: {}'.format(lim_factor))
                    
        return df_copy
    
    def make_object_schema(self):
        ''' 
        Makes a schema of object parts with their types and sizes.
        Returns object_schema = {
            part_type: [
                {'target':
                    {
                        'h': object-related height
                        'w': object-related width
                        'x': object-related center x
                        'y': object-related center y
                        'angle': degree of rotation
                    }
                },
                {...}
            ],
            ...
        }
        For some modes can contain also 'orig', not only 'target' coordinates.
        '''            
        mode = self.project_config['sampling']['object_schema_mode']
        if mode == 'random':
            object_schema = self.sample_part_features_random()
        elif mode == 'random_from_data':
            object_schema = self.sample_part_features_copy()
        elif mode == 'greedy_skeleton':
            object_schema = self.sample_part_features_copy()
        else:
            raise ValueError('mode {} id not supproted'.format(mode))
        
        return object_schema

    def sample_object_component_numbers(self, must_contain=[1], attempts=100, max_parts_per_object=12):
        for _ in range(attempts):
            object_components_dict = {}
            part_num_group = self.part_df.groupby(['class_type', 'obj_id'])['class_type'].count().groupby('class_type')
            for class_type in self.part_df['class_type'].unique():
                p_exists = self.part_df[self.part_df['class_type'] == class_type]['class_type'].count() / len(self.part_df)
                if random.random() < p_exists:
                    part_samples = int(max(np.random.normal(part_num_group.mean()[class_type], part_num_group.std()[class_type]), 0))
                    if part_samples > 0:
                        object_components_dict[class_type] = part_samples
                        
            if len(must_contain) == 0:
                return object_components_dict
            else:
                has_all = True
                for part_type in must_contain:
                    if part_type not in object_components_dict:
                        has_all = False
                if has_all and (sum(object_components_dict.values()) <= max_parts_per_object):
                    return object_components_dict
                
        return {1: 3}
    
    def sample_part_features_copy(self, real_plant=None):
        object_schema = {}
        
        if real_plant is None:
            real_random_obj_name = random.choice(self.part_df['obj_id'].unique())
            real_plant = self.part_df[self.part_df['obj_id'] == real_random_obj_name]
        
        for part_type in real_plant['class_type'].unique():
            parts_list = list()
            for part_idx, stats_row in real_plant[real_plant['class_type'] == part_type].iterrows():
                parts_list.append({
                    'target_coords': {
                        'h': stats_row['main_diag_height'],
                        'w': stats_row['main_diag_width'],
                        'x': stats_row['x_coord_object'],
                        'y': stats_row['y_coord_object'],
                        'angle': stats_row['alpha_horizons'],
                        'real_plant': stats_row # TODO: del
                    },
                })
                
            object_schema[part_type] = parts_list
        return object_schema
        
        
    def sample_part_features_random(self):
        object_schema = {}
        
        # Determine # of parts and their types
        object_components_dict = self.sample_object_component_numbers()

        for part_type in object_components_dict:
            parts_list = list()
            n_parttype_samples = object_components_dict[part_type]

            h_pdf = arr2hist(self.part_df['part_height'])
            sampled_h_list = sample_pdf(h_pdf, size=n_parttype_samples, add_noise=True)

            w_pdf = arr2hist(self.part_df['part_width'])
            sampled_w_list = sample_pdf(w_pdf, size=n_parttype_samples, add_noise=True)

            x_pdf = arr2hist(self.part_df['x_coord_object'])
            sampled_x_list = sample_pdf(x_pdf, size=n_parttype_samples, add_noise=True)

            y_pdf = arr2hist(self.part_df['y_coord_object'])
            sampled_y_list = sample_pdf(y_pdf, size=n_parttype_samples, add_noise=True)

            a_pdf = arr2hist(self.part_df['alpha_horizons'])
            sampled_a_list = sample_pdf(a_pdf, size=n_parttype_samples, add_noise=True)

            for i in range(n_parttype_samples):
                real_object_part = self.part_df[self.part_df['class_type'] == part_type].sample().iloc[0]
                
                parts_list.append({
                    'target_coords': {
                        'h': sampled_h_list[i],
                        'w': sampled_w_list[i],
                        'x': sampled_x_list[i],
                        'y': sampled_y_list[i],
                        'angle': sampled_a_list[i],
                        'real_plant': real_object_part
                    }
                })

            object_schema[part_type] = parts_list
        return object_schema
    
    def sample_objects_num(self):
        mode = self.project_config['sampling']['num_objects_per_image']
        
        if mode == 'random_from_data':
            num_obj_pdf = arr2hist(self.object_df.groupby('img_id')['tag_nums'].nunique())
            num_obj = max(int(sample_pdf(num_obj_pdf, size=1, add_noise=True)[0]), 1)
            return num_obj
        elif isinstance(mode, int):
            return mode
        
        return 1
                
    @staticmethod
    def is_complex_objects(row, object_prefix):
        return object_prefix in str(row['tag_nums'])
    
    @staticmethod
    def separate_complex_and_simple(orig_df, object_prefix='plant_'):
        df = orig_df.copy(deep=True)
        df['is_complex_object'] = df.apply(lambda row: FeatureSampler.is_complex_objects(row, object_prefix=object_prefix), axis=1)
        complex_df = df[df['is_complex_object']]
        simple_df = df[~df['is_complex_object']]
        
        if len(complex_df) == 0:
            complex_df = orig_df
        
        return complex_df, simple_df
