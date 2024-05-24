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


        self.case_list = load_json(self._base_dir+'dataset.json')[self.split]

        if num is not None:
            self.case_list = self.case_list[:num]

        print("--- total {} samples ---".format(len(self.case_list)))

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        filename=self.case_list[idx]

        dose = sitk.ReadImage(self._base_dir + 'gt/' + filename+'.nii.gz')
        dose = sitk.GetArrayFromImage(dose)
        target =  dose/70 #
        target = target[np.newaxis,:]

        image = self.load_img(filename)
        list_image = [image,target]
        # print('image.shape', image.shape)

        if self.transform:
            list_image = self.transform(list_image)

        sample = {'image': list_image[0], 'target': list_image[1]}
        if self.split=='val':
            sample['filename']=filename
        return sample

    def load_img(self,filename):
        images =sitk.ReadImage(self._base_dir + 'images/'+filename+'.nii.gz')
        images = sitk.GetArrayFromImage(images)
        ct=images[0,:,:]
        oar=images[1, :, :]
        ptv=images[2, :, :]
        dose_crt = images[3, :, :]

        ct = ct[np.newaxis, :]
        oar=oar[np.newaxis, :]
        ptv=ptv[np.newaxis, :]
        dose_crt = dose_crt[np.newaxis, :]
        dose_crt = dose_crt / 70  # 除70Gy,归一化

        ct = (ct-780)/440

        image = np.concatenate((ct,ptv,oar,dose_crt), axis=0) # 4 slices

        return image

