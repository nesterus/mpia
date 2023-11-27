import pandas as pd
import json
import os


def add_class_name(path_annotation, df_path):
    df = pd.read_csv(df_path, names=['file_name', 'class'])
    for idx, name in enumerate(os.listdir(path_annotation)):
        df_ = df[df['file_name'] == name[:-5]]
        try:
            new_tag_cl = {
                "name": 'class_' + list(df_['class'])[0]
            }
        except:
            new_tag_cl = {}

        f = open(os.path.join(path_ann, name))

        data = json.load(f)
        objects = data['objects']
        for obj in objects:
            tags = obj['tags']
            tags.append(new_tag_cl)
            obj['tags'] = tags

            with open(os.path.join(path_ann, name), "w") as write_file:
                json.dump(data, write_file)
