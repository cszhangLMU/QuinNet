import os

import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        
        mask = []
        dataset_name = self.img_dir.split('/')[-2]
        if dataset_name == 'ISIC2018':
            for i in range(self.num_classes):
                ##mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                            ##img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
                mask.append(cv2.imread(os.path.join(self.mask_dir,
                            img_id + '_segmentation'  + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        elif dataset_name == 'ISIC2018NEW2':
            for i in range(self.num_classes):
                ##mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                            ##img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
                mask.append(cv2.imread(os.path.join(self.mask_dir,
                            img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        elif dataset_name == 'BUSI':    
            for i in range(self.num_classes):
                mask_list = glob(os.path.join(self.mask_dir, img_id + '_mask' + '*' + self.mask_ext))
                #print('=========================', len(mask_list))
                #print(mask_list)
                mask_img = cv2.imread(mask_list[0], cv2.IMREAD_GRAYSCALE)
                if len(mask_list)>1:  # 如果存在多张mask图像，将多张合并为一张mask图像
                    for k in range(len(mask_list)): 
                        mask_img = cv2.add(mask_img, cv2.imread(mask_list[k], cv2.IMREAD_GRAYSCALE))
                mask.append(mask_img)
        elif dataset_name == 'ICF':
            for i in range(self.num_classes):
                mask.append(cv2.imread(os.path.join(self.mask_dir,
                            img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        else :#dataset_name == 'cvc-clinicdb   和COVID-19'  和Kvasir-SEG  和kvasir-instrument
            for i in range(self.num_classes):
                mask.append(cv2.imread(os.path.join(self.mask_dir,
                                                    img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[
                                ..., None])
        mask = np.dstack(mask)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
        
class Data_loaderVE(Dataset):
    def __init__(self, root, train, transform=None):

        self.train = train  # training set or test set
        self.data, self.y, self.y2 = torch.load(os.path.join(root, train))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1, y1, y2 = self.data[index], self.y[index], self.y2[index]  # y1: Lung, y2: Infection      

        img1 = np.array(img1)
        y1 = np.array(y1)
        y2 = np.array(y2)
        img1 = img1.astype(np.uint8) 
        y1 = y1.astype(np.uint8) 
        y2 = y2.astype(np.uint8)                

        y1[y1 > 0.0] = 1.0
        y2[y2 > 0.0] = 1.0
        y33 = y2*255.0
        y33 = y33.astype(np.uint8)

        kernel = np.ones((5,5),np.uint8)
        y3 = cv2.morphologyEx(y33, cv2.MORPH_GRADIENT, kernel)


        y3[y3 > 0.0] = 1.0    
        if self.transform is not None:
            augmentations = self.transform(image=img1, masks=[y1, y2, y3])

            image = augmentations["image"]
            mask = augmentations["masks"][0]
            mask1 = augmentations["masks"][1]#肺部感染
            edge = augmentations["masks"][2]

            
        return   image, mask1, mask  # return   image, mask, mask1

    def __len__(self):
        return len(self.data)
