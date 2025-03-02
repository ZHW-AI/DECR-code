import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import time

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split



import albumentations as A
from albumentations.pytorch import ToTensorV2

#==============private DATA
from DataSet.data_distill import Dataset_name  
#==============Private Model
from Model.cenet import CE_Net_OCT
#===========train loop
from Loop.framwork import loop
#=======early stop ==================
from util.early_stop import EarlyStopping
from util.checkpoint_save import best_checksave
from util.vis_data import vis_classifi

#==============SEED==================================
import random
seed = 42
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
# 设置样本shuffle随机种子，作为DataLoader的参数
g.manual_seed(0)

#=================hyperparameters,===================================

class argparse():
    pass

args = argparse()

args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]


#==========model,loss,optimizers===========
from Model.TS_Net_AE import AE_T_r18_S_r18_
# from Model.TS_Net import TS_tr18_sr18 #注释掉ImportError: cannot import name 'swin_b' from 'torchvision.models'
args.model = AE_T_r18_S_r18_()
from util.Distill_loss import mse_cosSimilar #cosSimilar
args.criterion = mse_cosSimilar()
# from util.metirc import Dice_gpu,IoU_gpu
from sklearn.metrics import roc_auc_score

args.metrics=[roc_auc_score]
args.metrics_check=roc_auc_score#metric_gpu
args.metrics_max =True

args.epochs, args.learning_rate, args.patience = [500, 3e-4, 4]
args.optimizer = torch.optim.Adam(args.model.parameters(),lr=args.learning_rate)
args.milestones = [20,100,200]
args.scheduler = torch.optim.lr_scheduler.MultiStepLR(args.optimizer, args.milestones, gamma=0.5, last_epoch=-1)
args.best_checksave = best_checksave(verbose=True,save_param=True,save_max=args.metrics_max)
# args.early_stopping = EarlyStopping(patience=args.patience,verbose=True)

args.pixel_label = True
# args.train_root_path = "/home/lus/myProject/data_source/MahalanobisAD-pytorch-master/data/RNFLdetection/RNFLD"#'../../datasets/OCT2017_DME'
# args.val_root_path = "/home/lus/myProject/data_source/MahalanobisAD-pytorch-master/data/RNFLdetection/RNFLD"#'../../datasets/OCT2017_DME'
# args.test_root_path ="/home/lus/myProject/data_source/MahalanobisAD-pytorch-master/data/RNFLdetection/RNFLD" #'../../datasets/OCT2017_DME'

args.class_name = 'SL13'
args.train_root_path = os.path.join('/data/students/Share_files/YDFID-20220311/' ,args.class_name)
args.val_root_path = os.path.join('/data/students/Share_files/YDFID-20220311/' ,args.class_name)
args.test_root_path = os.path.join('/data/students/Share_files/YDFID-20220311/' ,args.class_name)

# args.train_root_path = "/data/students/Share_files/YDFID-20220311/CL13"#"data/Covid19-dataset"#"data/Brain-Tumor-MRI-Dataset"#OCT2017_FULL_Test"#'../../datasets/OCT2017_DME'
# args.val_root_path = "/data/students/Share_files/YDFID-20220311/CL13"#"data/Covid19-dataset"#"data/Brain-Tumor-MRI-Dataset"#OCT2017_FULL_Test"#'../../datasets/OCT2017_DME'
# args.test_root_path ="/data/students/Share_files/YDFID-20220311/CL13"#"data/Covid19-dataset"#"data/Brain-Tumor-MRI-Dataset"#OCT2017_FULL_Test" #'../../datasets/OCT2017_DME'
# # /data/students/Share_files/YDFID-20220311

args.img_size = [224,224]
args.batch_size=2
args.num_workers = 8

#vis
args.visualization =[] #'visdom','tensorboardX'
args.visdom_port = 8098
args.env_add = ''
args.env = args.train_root_path.split("/")[-1] +"_"+ type(args.model).__name__ + args.env_add
train_localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
train_localday = time.strftime("%Y-%m-%d", time.localtime())

args.model_load_path = os.path.join('checkpoints/',args.env)
args.model_save_path = os.path.join(args.model_load_path,train_localday,train_localtime)
args.results_save_path = os.path.join('results/',args.env)
args.vis_data = vis_classifi(args.visualization,env=args.env,
                metrics= args.metrics,visdom_port=args.visdom_port,model_save_path= args.model_save_path)



#===========Dataset,DataLoader==============
# Data augmentation and normalization for training
# Just normalization for valation

args.data_transforms = {
    'train': A.Compose([
        
        # A.Flip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(args.img_size[0], args.img_size[1]),
    ]),
    'val': A.Compose([A.Resize(args.img_size[0], args.img_size[1])]),
    'test': A.Compose([A.Resize(args.img_size[0], args.img_size[1])]),
}



train_dataset = Dataset_name(flag='train', root_path=args.train_root_path,
    data_transforms=args.data_transforms,class_name=args.class_name)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,worker_init_fn=seed_worker,generator=g,)

val_dataset = Dataset_name(flag='val', root_path=args.val_root_path,
    data_transforms=args.data_transforms,pixel_label=args.pixel_label,class_name=args.class_name)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,num_workers=args.num_workers,worker_init_fn=seed_worker,generator=g,)

test_dataset = Dataset_name(flag='test', root_path=args.test_root_path,
    data_transforms=args.data_transforms,pixel_label=args.pixel_label,class_name=args.class_name)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,num_workers=args.num_workers,worker_init_fn=seed_worker,generator=g,)

#======
Net_work = loop(args)

modeis_train = True
if modeis_train:
    #============================train and val =========================
    Net_work(args,train_dataloader,val_dataloader,test_dataloader)
else:
    #============================ Pred =================================
    model_load_path  = 'checkpoints/chest_xray_AE_T_r18_S_r18_/2022-11-02/2022-11-02_01-50-19/chest_xray_AE_T_r18_S_r18__val_best.pth'
    print(f"loading =====> {model_load_path}")
    model_device = args.model.to(args.device)
    model_device.load_state_dict(torch.load(model_load_path))

    Net_work.test_loop(test_dataloader=test_dataloader,model=model_device,args=args)
    """
    Net_work.test_loop参数说明,如果想预测验证集 使用如下参数 test_dataloader=val_dataloader
                                如果想预测测试集 使用如下参数 test_dataloader=test_dataloader
    """



    