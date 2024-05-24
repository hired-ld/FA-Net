from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json



class dataset_load(Dataset):
    """  dataset """
    def __init__(self, base_dir=None, split='train', num=None,  transform=None,beam=False,beam_d=False,ptv_d=False):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.split = split


        if self.split == 'val':
            self.case_list = load_json(self._base_dir+'dataset.json')[self.split]
        else:
            self.case_list = os.listdir(self._base_dir+'gt/')

        if num is not None:
            self.case_list = self.case_list[:num]

        print("--- total {} samples ---".format(len(self.case_list)))

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        filename=self.case_list[idx]
        if self.split == 'val':
            dose = sitk.ReadImage(self._base_dir + 'gt/' + filename+'.nii.gz')
        else:
            dose = sitk.ReadImage(self._base_dir + 'gt/' + filename)
        dose = sitk.GetArrayFromImage(dose)
        target =  dose/70 #
        target = target[np.newaxis,:]

        image = target
        list_image = [image,target,target]
        # print('image.shape', image.shape)

        if self.transform:
            list_image = self.transform(list_image)

        sample = {'image': list_image[0], 'target': list_image[1]}
        if self.split=='val':
            sample['filename']=filename
        return sample
