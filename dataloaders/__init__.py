import torch.utils.data as data
import numpy as np
import torch
import random
import cv2
from .DataAugmentation import random_flip_2d,  random_rotate_around_z_axis, random_translate, to_tensor

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(2022)

def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_2d(list_images, list_axis=[1], p=1.0)

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angle=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_boder_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,
                                   roi_mask=list_images[1][0, :, :],  # the possible dose mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])

    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images

def make_data_loader(args, **kwargs):
    if args.model == 'mynet':
        from dataloaders.liver_dose_2d import dataset_load
        root_path = '../dose_datasets/'+args.dataset+'/reprocess_data/'
    elif args.model  == 'AENet':
        from dataloaders.liver_dose_2d_aenet import dataset_load
        root_path = '../dose_datasets/' + 'liver_dose3' + '/reprocess_data/'


    train_dataset = dataset_load(base_dir=root_path,
                                 split='train',
                                 transform=train_transform
                                 )
    val_dataset = dataset_load(base_dir=root_path,
                               split='val',
                               transform=val_transform
                               )

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,**kwargs)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10,**kwargs)

    return train_loader, val_loader


