import albumentations as A

deafult_border_mode = 0
deafult_interpolation = 4

aug_list = [
    {
        'target': ['background'],  # background object object_part
        'terminate': False,  # If this augmentation applied, other are not considered
        'apply_to_classes': 'all',  # 'all' or list
        'apply_to_object_types': 'all',
        'apply_to_tags': 'all',
        'exclude_classes': [],
        'exclude_object_types': [],
        'exclude_tags': [],
        'pipeline':
            A.Compose([
                A.CLAHE(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75, border_mode=deafult_border_mode, interpolation=deafult_interpolation),
                A.Blur(blur_limit=3),
                A.OpticalDistortion(border_mode=deafult_border_mode, interpolation=deafult_interpolation),
                A.GridDistortion(border_mode=deafult_border_mode, interpolation=deafult_interpolation),
                A.HueSaturationValue(),
            ])
    },
    {
        'target': ['object_part'],  # background object object_part
        'terminate': False,  # If this augmentation applied, other are not considered
        'apply_to_classes': 'all',  # 'all' or list
        'apply_to_object_types': 'all',
        'apply_to_tags': 'all',
        'exclude_classes': [],
        'exclude_object_types': [],
        'exclude_tags': [],
        'pipeline':
            A.Compose([
                A.CLAHE(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75, border_mode=deafult_border_mode, interpolation=deafult_interpolation),
                A.CoarseDropout(p=1),
                A.ISONoise(p=1),
                A.Blur(blur_limit=5),
                A.OpticalDistortion(border_mode=deafult_border_mode, interpolation=deafult_interpolation),
                A.GridDistortion(border_mode=deafult_border_mode, interpolation=deafult_interpolation),
                A.HueSaturationValue(),
            ])
    },
]

