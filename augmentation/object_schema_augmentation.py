import random


def augment_object_schema(parts_data, max_angle=10):
    for part_type in parts_data:
        for part_idx, part in enumerate(parts_data[part_type]):
            parts_data[part_type][part_idx]['target_coords']['angle'] = int(parts_data[part_type][part_idx]['target_coords']['angle'] + int(random.random() * max_angle) - max_angle // 2)
            
    return parts_data
