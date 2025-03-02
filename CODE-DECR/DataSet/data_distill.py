from cProfile import label
import os
from tkinter import image_names

import pandas
import cv2
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import glob

class Dataset_name(Dataset):
    def __init__(self, flag='train', root_path='',data_transforms=None,threshold=0.5,pixel_label=False,class_name='CL1'):
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.root_path = root_path
        self.transform = data_transforms
        self.threshold= threshold
        self.pixel_labels = pixel_label#xiugaimingzi
        self.class_name = class_name
        print("="*10+"Data setting"+"="*10)
        print(f"Label value threshold : {self.threshold}")

        # self.images,self.masks = self.read_Anomaly_Detection(dataset_root = self.root_path,
                                    # mode =self.flag)
        self.images,self.masks,self.labels,self.names = self.read_Anomaly_Detection(dataset_root = self.root_path, mode =self.flag)
        

    def __getitem__(self, idx):
    
        transform = self.transform
        mode = self.flag      

        assert mode in ["train","val","test"]
        if mode == 'train':
            image_path,names_df = self.images[idx],self.names[idx]
            img = cv2.imread(image_path).copy() # [:, :, ::-1] BGR -> RGB
            transformed = transform[mode](image=img)
            transformed_image = transformed['image']
            transformed_image = (transformed_image/255.0).astype(np.float32)#.to(torch.float32)
            transformed_image = transformed_image.transpose([2,0,1]) #H,W,C->C,H,W
            transformed_image = torch.tensor(transformed_image)#numpy -> tensor
            return transformed_image, [],names_df #image_path.split('/')[-1]去掉返回名字

        else:# "test","val"
            if self.pixel_labels == True:#xiugaimingzi
                images,masks,labels,image_names = self.pixel_label(idx=idx,mode=mode)
            else:
                images,masks,labels,image_names = self.sample_label(idx=idx,mode=mode)
            return images,masks,labels,image_names



    def sample_label(self,idx,mode):
        mode = self.flag

        image_path, label_pos = self.images[idx],self.masks[idx]

        img = cv2.imread(image_path).copy() # [:, :, ::-1] BGR -> RGB
        HW = torch.tensor(img.shape)


        transformed = self.transform[mode](image=img)
        transformed_image = transformed['image']


        transformed_image = (transformed_image/255.0).astype(np.float32)#.to(torch.float32)
        label = np.array(label_pos,dtype=np.float32)#.astype(np.float32)#.to(torch.float32)
        
        transformed_image = transformed_image.transpose([2,0,1]) #H,W,C->C,H,W

        transformed_image = torch.tensor(transformed_image)#numpy -> tensor
        label = torch.tensor(label)#numpy -> tensor
        
        if mode == "val":
            return transformed_image,label,image_path.split('/')[-1]
        elif mode == "test":
            return transformed_image,HW,image_path.split('/')[-1]


    def pixel_label(self,idx,mode):
        mode = self.flag
    
        # image_path, label_pos = self.images[idx],self.masks[idx]
        image_path, masks,labels,names = self.images[idx],self.masks[idx],self.labels[idx],self.names[idx]

        img = cv2.imread(image_path).copy() # [:, :, ::-1] BGR -> RGB
        HW = torch.tensor(img.shape)   #将 Numpy 数组转换为 PyTorch 张量的方法 输出: tensor([H, W]) 原始图像的
        if labels == 0:
            # label = np.zeros([1, img.shape[0], img.shape[1]])
            gt_mask = np.zeros([img.shape[0], img.shape[1],1]) #(512, 512,1)

        else:
            # label_path = os.path.join(self.root_path,"/GroundTruth/defect")
            gt_mask = cv2.imread(masks,cv2.IMREAD_GRAYSCALE).copy() # [:, :, ::-1] BGR -> RGB  cv2.IMREAD_GRAYSCALE 参数表示以灰度模式读取图像
            # gt_mask = cv2.imread(masks).copy()
            gt_mask = np.expand_dims(gt_mask,axis= 2)#将单通道的灰度图像转换为与彩色图像相同的通道数，以便与模型进行输入和处理
        
        # print('img',img.shape)#img (512, 512, 3)
        # print('gt_mask',gt_mask.shape)#(512, 512,1)
        transformed = self.transform[mode](image=img,mask=gt_mask)
        transformed_image = transformed['image']#shape:(224, 224, 3)
        transformed_masks = transformed['mask']#shape:(224, 224, 1)

        transformed_image = (transformed_image/255.0).astype(np.float32)#.to(torch.float32)
        transformed_masks = (transformed_masks/255.0).astype(np.float32)#.to(torch.float32)
# 因为transformed_masks里面是由0，1组成的，所以1除以255后，肯定比0.5小，所以在下一行代码中所以的1将全变成0；
        transformed_masks[transformed_masks >= self.threshold] = 1.0
        transformed_masks[transformed_masks <  self.threshold] = 0.0
        
        transformed_image = transformed_image.transpose([2,0,1]) #H,W,C->C,H,W
        transformed_masks = transformed_masks.transpose([2,0,1]) #H,W,C->C,H,W
        transformed_image = torch.tensor(transformed_image)#numpy -> tensor
        transformed_masks = torch.tensor(transformed_masks)#numpy -> tensor
        # transformed_masks = torch.unsqueeze(transformed_masks,dim=0) #H,W->1,H,W
        
        if mode == "val":
            # return transformed_image,transformed_masks,image_path.split('/')[-1]
            return transformed_image,transformed_masks,labels,names
        
        elif mode == "test":
            return transformed_image,transformed_masks,HW,names#image_path.split('/')[-1]
        
        pass
    
    def __len__(self):
        return len(self.images)
    
    # def read_Anomaly_Detection(self,dataset_root = '',mode = 'test'):
    #     assert mode in ["train","val","test"],"three mode can pass, other ones are permited "
    #     images = []
    #     labels  = []
    #     if mode == "train":
    #         image_root = os.path.join(dataset_root,'train/defect-free')
    #         images = [os.path.join(image_root, image_name)  for image_name in os.listdir(image_root)]
    #         # gt_root = os.path.join(dataset_root,'label')
    #     else:
    #         for dir_path in os.listdir(os.path.join(dataset_root,"test")):
    #             image_root = os.path.join(dataset_root,"test",dir_path)
    #             single_path = [os.path.join(image_root, image_name)  for image_name in os.listdir(image_root)]
    #             images.extend(single_path)

    #             if dir_path == "defect-free":
    #                 image_root = os.path.join(dataset_root,"test",dir_path)
    #                 single_label = [0  for image_name in os.listdir(image_root)]
    #             else:
    #                 image_root = os.path.join(dataset_root,"test",dir_path)
    #                 single_label = [1  for image_name in os.listdir(image_root)]

    #             labels.extend(single_label)

    #     print('{}: number of data is {}'.format(mode,len(images)) )
    #     return images, labels
    
    # 修改后的数据读取，增加了gt_mask和name  
    def read_Anomaly_Detection(self,dataset_root = '',mode = 'test'):
        assert mode in ["train","val","test"],"three mode can pass, other ones are permited "
        images = []
        labels  = []
        mask = [] 
        name = [] 
        if mode == "train":
            image_root = os.path.join(dataset_root,'train/defect-free')#/data/students/Share_files/YDFID-20220311/SP24/train/defect-free
            # images = [os.path.join(image_root, image_name)  for image_name in os.listdir(image_root)]
            img_paths = glob.glob(os.path.join(image_root + "/*.png"))
            img_paths.sort()#排序
            images.extend(img_paths)
            labels.extend([0] * len(img_paths))
            mask.extend([0] * len(img_paths))
            # name.extend(['defect-free'] * len(img_paths))
            na = [f'{self.class_name}_defect-free_{os.path.basename(s)[:-4] }'for s in img_paths]
            name.extend(na)
        else:
            img_dir = os.path.join(dataset_root,"test")#/data/students/Share_files/YDFID-20220311/SP24/test
            gt_dir = os.path.join(dataset_root,"GroundTruth")#/data/students/Share_files/YDFID-20220311/SP24/GroundTruth
            defect_types = sorted(os.listdir(img_dir))

            for defect_type in defect_types:
                if defect_type == 'defect-free':
                    img_paths = glob.glob(os.path.join(img_dir, defect_type) + "/*.png")#0:'/data/Users/menglp/myProject/DataSet/YDFID/SL0/train/defect-free/000.
                    img_paths.sort()#排序
                    images.extend(img_paths)
                    labels.extend([0] * len(img_paths))
                    mask.extend([None] * len(img_paths))
                    # name.extend(['defect-free'] * len(img_paths))
                    # name = [os.path.join(self.class_name, defect_type,os.path.basename(s)[:-4]) for s in img_paths]
                    na = [f'{self.class_name}_{defect_type}_{os.path.basename(s)[:-4] }'for s in img_paths]
                    name.extend(na)

                else:
                    img_paths = glob.glob(os.path.join(img_dir, defect_type) + "/*.png")#0:'/data/Users/menglp/myProject/DataSet/YDFID/SL0/test/defect/000.png'
                    gt_paths = [os.path.join(gt_dir, defect_type, os.path.basename(s)[:-4] + '_mask.png') for s in img_paths]
                    #0:'/data/Users/menglp/myProject/DataSet/YDFID/SL0/GroundTruth/defect/000_mask.png'
                    img_paths.sort()#排序
                    gt_paths.sort()#排序
                    images.extend(img_paths)
                    mask.extend(gt_paths)
                    labels.extend([1] * len(img_paths))
                    na = [f'{self.class_name}_{defect_type}_{os.path.basename(s)[:-4] }'for s in img_paths]#列表生成式
                    name.extend(na)



        print('{}: number of data is {}'.format(mode,len(images)) )
        assert len(images) == len(labels), "Something wrong with test and ground truth pair!"
        return images, mask, labels, name#test中defect在前，defect-free在后



