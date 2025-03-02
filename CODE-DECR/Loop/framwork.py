from builtins import print
from cProfile import label
from unicodedata import name
import os
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm

import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from util.metrics import *
from util.visualization import *

def denormalization(x):
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        # x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
        x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)

        return x #shape:(224, 224, 3)

class loop():
    def __init__(self,args) -> None:
        self.env =args.env#visdom env
        
        #save model time
        train_localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        train_localday = time.strftime("%Y-%m-%d", time.localtime())

        self.model_load_path = args.model_load_path
        self.model_save_path = args.model_save_path
        self.results_save_path = args.results_save_path

        #log model paramters metrics save
        self.log_save_path = self.model_save_path
        #mkdirs path of results and model


        if os.path.exists(self.results_save_path):
            shutil.rmtree(self.results_save_path)
            
        if not os.path.exists(self.results_save_path):
            os.makedirs(self.results_save_path)
        if not os.path.exists(os.path.join(self.results_save_path,'vis')):
            os.makedirs(os.path.join(self.results_save_path,'vis'))
            os.makedirs(os.path.join(self.results_save_path,'anomap'))
            os.makedirs(os.path.join(self.results_save_path,'error'))

    def __call__(self,args,train_dataloader,val_dataloader,test_dataloader):
        # create path of model and log
        if not os.path.exists(self.model_save_path) :
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.log_save_path):
            os.makedirs(self.log_save_path)

        train_epochs_loss = []
        val_epochs_loss = []
        # train_epochs_metric = []
        val_epochs_metric = []
        learning_rate_epochs = []
        model_device = args.model.to(args.device)
        #====================train and adjust Learning rate
        self.Full_logger = Logg(self.log_save_path,log_name ='Full_log.txt')#save log  
        with open(os.path.join(self.log_save_path , 'option.py'), 'w') as  mylog:
            self.print_options(args,file=mylog)

        for epoch in range(args.epochs):
            
            self.Full_logger.logger.debug(f"Epoch {epoch+1}/{args.epochs}-------------------------------{args.class_name}")#print(f"Epoch {epoch+1}\n-------------------------------")
            #===========learning rate log
            learning_rate_epochs.append(args.optimizer.param_groups[0]['lr'])
            for param in args.optimizer.param_groups:
                self.Full_logger.logger.debug(f"learning rate {param['lr']}")#print(f"learning rate {param['lr']}")
            #=====================train========================
            train_log = self.train_loop(train_dataloader= train_dataloader,model =model_device, criterion= args.criterion,optimizer=args.optimizer,epoch = epoch,args=args)
            #=====================val============================
            val_log = self.val_loop(val_dataloader,model_device,args.criterion,args)
            #=====================Learning rate adjust=============
            args.scheduler.step()
            #=====================best_model save and pred=================
            val_metric = val_log['metrics_value'][args.metrics_check.__name__]
            update_point = args.best_checksave(val_metric=val_metric,model=model_device,path=self.model_save_path,
                                                            checkpoint_name=self.env+'_val_best.pth',logger=self.Full_logger)#save point
            # if update_point : self.test_loop(test_dataloader, model_device,args)#best pred
            self.test_loop(test_dataloader, model_device,args)
            
            #====================last epoch pred==================
            # if (epoch+1)==args.epochs: self.test_loop(test_dataloader, model_device,args)
            #====================vis and log =====================
            # args.vis_data(train_log,val_log)
            train_epochs_loss.append(train_log['train_epoch_avg_loss'])
            val_epochs_loss.append(val_log['val_epoch_avg_loss'])
            # train_epochs_metric.append(train_log["metrics_value"][args.metrics_check.__name__])
            val_epochs_metric.append(val_log["metrics_value"][args.metrics_check.__name__])
            # #==================early stopping======================
            # args.early_stopping(val_epoch_avg_loss[-1],model=args.model,path=r'./model_to_save')
            # if args.early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            # #====================adjust lr=========================
            # lr_adjust = {
            #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            #         10: 5e-7, 15: 1e-7, 20: 5e-8
            #     }
            # if epoch in lr_adjust.keys():
            #     lr = lr_adjust[epoch]
            #     for param_group in args.optimizer.param_groups:
            #         param_group['lr'] = lr
            #     print('Updating learning rate to {}'.format(lr))

        #============================绘图 ==========================
        plt.figure(figsize=(12,8))
        plt.subplot(211)
        # plt.plot(train_epochs_metric[:],':D',label="train_metric")
        plt.plot(val_epochs_metric[:],':D',label="val_metric")
        plt.legend()
        plt.title("metric")
        plt.subplot(212)
        plt.plot(train_epochs_loss[1:],':D',label="train_loss")
        plt.plot(val_epochs_loss[1:],':D',label="val_loss")
        plt.title("epochs_loss")
        plt.legend()
        # plt.show()      
        plt.savefig(os.path.join(self.model_save_path,"metirc_loss.png"))  

        plt.figure(figsize=(6,4))
        plt.plot(learning_rate_epochs[:],'-D',label="learning rate")
        plt.title("learning rate")
        plt.savefig(os.path.join(self.model_save_path,"learning-rate.png")) 

    def train_loop(self,train_dataloader, model, criterion, optimizer,epoch,args):
        model.train()
        ###修改
        model.T_block1.eval()
        model.T_layer1.eval()
        model.T_layer2.eval()
        model.T_layer3.eval()
        model.T_layer4.eval()

        metrics_estimate = metric_eval(metrics=args.metrics)
        metrics_images = {m.__name__:[] for m in metrics_estimate.metrics}
        train_epoch_loss = []
        train_cache_gpu = []

        with tqdm(train_dataloader,desc="Train") as pbar:
            for idx,(data_x,_,data_name) in enumerate(pbar,0):
                data_x = data_x.to(args.device)
                # data_y = data_y.to(args.device)
                outputs = model(data_x)
                
                loss = criterion(outputs[0],outputs[1])
                # loss = criterion(outputs[0],outputs[2]) + criterion(outputs[1],outputs[2]) + 10*criterion(outputs[0],outputs[1])
                #-----
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #-----

                train_epoch_loss.append(loss.item())

                # metircs_batches = metrics_estimate([outputs.detach(),data_y.detach()])

                # for k in metrics_images.keys(): 
                #     metrics_images[k].extend(metircs_batches[k].reshape([-1,1])) 
                
                # tqdm_log2 = {k:'{0:1.5f}'.format(np.average(metircs_batches[k])) for k in metrics_images.keys()}
                tqdm_log1 = {'loss' : '{0:1.5f}'.format(loss.item())}
                # tqdm_log1.update(tqdm_log2)
                pbar.set_postfix(tqdm_log1) #输入一个字典，显示实验指标

        # metrics_value = {k:np.average(metrics_images[k])
        #                             for k in metrics_images.keys()}
        # for k,v in metrics_value.items():
        #     self.Full_logger.logger.debug("train metric {}:{}".format(k,v))#print("train metric {}:{}".format(k,v))
        # print(f"train metric value {metrics_value}")
        train_epoch_avg_loss = np.average(train_epoch_loss)
        get_N = min(4,data_x.size(0))
        return {'epoch':epoch,'train_epoch_avg_loss':train_epoch_avg_loss,
            'inputs':data_x[:get_N].detach()}



    def val_loop(self,val_dataloader, model,criterion,args):
        model.eval()
        metrics_estimate = metric_eval(metrics=args.metrics)
        metrics_images = {m.__name__:[] for m in metrics_estimate.metrics}
        val_epoch_loss = []

        gt_list_px = []
        pr_list_px = []
        gt_list_sp = []
        pr_list_sp = []
        aupro_list = []
        
        #新添加
        gt_label_list = []
        gt_mask_list = []
        scores = None

        with torch.no_grad():
            for idx,(data_x,gt_mask,data_y,data_name) in enumerate(tqdm(val_dataloader,desc="Valid"),0):

                gt_label_list.extend(data_y.cpu().numpy())

                for i in range(gt_mask.shape[0]):
                    gt_mask_list.append(gt_mask[i].squeeze().cpu().numpy())

                data_x = data_x.to(args.device)#torch.Size([1, 3, 224, 224])img = cv2.imread(image_path).copy()
                gt_mask = gt_mask.to(args.device)
                data_y = data_y.to(args.device)

                outputs = model(data_x)

                loss = criterion(outputs[0],outputs[1])
                val_epoch_loss.append(loss.item())


                anomaly_map, _ = cal_anomaly_map(outputs[0], outputs[1],out_size = args.img_size, amap_mode='a')#shape:(224, 224)
                anomaly_map = anomaly_map.astype(np.float32) # float64 转 32 
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)#shape:(224, 224)

                if scores is None:
                    scores = []

                scores.append(anomaly_map)

                #data_y.size=batchsize=1--大概率都不会被执行
                # if len(data_y.size()) > 2:
                #     data_y[data_y > 0.5] = 1
                #     data_y[data_y <= 0.5] = 0
                #     gt_list_px.extend(data_y.cpu().numpy().astype(int).ravel())
                #     pr_list_px.extend(anomaly_map.ravel())
                    
                #     if torch.max(data_y)!=0:
                #         aupro_list.append(compute_pro(data_y.squeeze(0).cpu().numpy().astype(int),
                #                                     anomaly_map[np.newaxis,:,:]))

                # gt_list_sp.append(np.max(data_y.cpu().numpy().astype(int)))#不需要max
                # pr_list_sp.append(np.max(anomaly_map))

            # auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
            # auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
            
            # aupro_px = round(np.mean(aupro_list),3)
            # threshold_f1 = return_best_thr(gt_list_sp, pr_list_sp)
            # f1_value,acc,SEN,SPE = compute_metrics(gt_list_sp, pr_list_sp,threshold=threshold_f1)

            # 计算评价指标
            cal_pro=True
            img_roc_auc, per_pixel_rocauc, pro_auc_score, threshold = metric_cal(np.array(scores), gt_label_list, gt_mask_list, cal_pro=cal_pro)
            
        # metrics_value = {'Pixel Auroc':auroc_px,'Sample Auroc':auroc_sp, "Pixel Aupro":aupro_px}
        # metrics_value = {'Sample Auroc':auroc_sp}
        # metrics_value = {roc_auc_score.__name__:auroc_sp,"F1":f1_value,"ACC":acc,"SEN":SEN,"SPE":SPE}
        metrics_value = {roc_auc_score.__name__:img_roc_auc,"per_pixel_rocauc":per_pixel_rocauc,"pro_auc_score":pro_auc_score,"threshold":threshold}
        

        for k,v in metrics_value.items():
            self.Full_logger.logger.debug("val   metric {}:{}".format(k,v))#print("val   metric {}:{}".format(k,v))
        
        get_N = min(4,data_x.size(0))
        val_epoch_avg_loss = np.average(val_epoch_loss)
        return {'val_epoch_avg_loss':val_epoch_avg_loss,"metrics_value":metrics_value,
            'inputs':data_x[:get_N].detach(),'labels':data_y[:get_N].detach()}
#测试 
    def test_loop(self,test_dataloader, model,args):
        model.eval()
        All_Preds = []

        #新添加
        test_imgs = []
        names = []
        gt_mask_list = []
        scores = None

        with torch.no_grad():
            for idx,(data_x,gt_mask,HW,data_name) in enumerate(test_dataloader,0):

                for i in range(gt_mask.shape[0]):
                    gt_mask_list.append(gt_mask[i].squeeze().cpu().numpy())

                for d, n in zip(data_x, data_name):
                    test_imgs.append(denormalization(d.cpu().numpy()))
                    names.append(n)

                data_x = data_x.to(args.device)
                
                outputs = model(data_x)

                anomaly_map, a_map_list = cal_anomaly_map(outputs[0], outputs[1],out_size = args.img_size, amap_mode='a')
                #(224, 224)
                anomaly_map = anomaly_map.astype(np.float32) # float64 转 32
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                if scores is None:
                    scores = []

                scores.append(anomaly_map)

                # concat_map = torch.cat(a_map_list,dim=-1)

                # for n,img,p,img_size in zip(data_name,data_x.cpu().detach().numpy(),[anomaly_map],HW):
                  
                #     p = min_max_norm(p)
                #     ano_map = cvt2heatmap(p*255)
                #     H,W =img_size[0].item(),img_size[1].item()
                #     # ano_map = cv2.resize(ano_map,(W,H))

                #     # cv2.imwrite(os.path.join(self.results_save_path,n),ano_map)
                #     vis_result = self.vis(img.transpose([1,2,0]),ano_map,label=None)
                #     cv2.imwrite( os.path.join(self.results_save_path ,'vis',n) ,vis_result)
                #     cv2.imwrite( os.path.join(self.results_save_path ,'anomap',n) ,ano_map)
                #     cv2.imwrite( os.path.join(self.results_save_path ,'error',n) ,(p*255).astype("uint8"))
            
        
            img_dir = os.path.join(self.results_save_path ,'heatmap')
            if not os.path.exists(img_dir) :
                os.makedirs(img_dir)
            plot_sample_cv2(names, test_imgs, {'CDO': scores}, gt_mask_list, save_folder=img_dir)
            plot_anomaly_score_distributions({'CDO': scores}, gt_mask_list, save_folder=img_dir,
                                                class_name=args.class_name)
        
        get_N = min(4,data_x.size(0))

        # return {'inputs':data_x[:get_N].detach(),'preds':outputs[:get_N].detach()}
        return {'inputs':data_x[:get_N].detach()}


    def print_options(self,class_,file):
        # message = ''
        # message += '----------------- Options ---------------\n'
        # for k, v in sorted(vars(opt).items()):
        #     comment = ''
        #     # message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        #     message += '{}: {}{}\n'.format(str(k), str(v), comment)
        # message += '----------------- End -------------------'    
        for k,v in class_.__dict__.items():
            print(k,"=",v,file=file)
            print('#'+'-'*40,file=file)

    def vis(self,img,pred,label=None):
    
        
        assert np.max(img)<=1 #and np.min(img)>=0
        img = (img *255).astype(np.uint8)
        img = cv2.UMat(img).get()


        origin = img.copy()

        img= show_cam_on_image(img, pred)




        if not label is None:
            label = label.astype(np.uint8)
            ret, binary = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(0,255,0),1)


        concate=np.concatenate((origin,img),axis=1)

        return concate



class metric_eval():
    def __init__(self,metrics) -> None:
        self.metrics =metrics
        if len(self.metrics) != 0:
            self.metrics_dict ={}
            # for m in self.metrics:
            #     self.metrics_dict[m.__name__] = []

    def __call__(self, cache_gpu):

        if len(self.metrics) != 0:
            for m in self.metrics:
                if 'gpu' not in m.__name__: 
                    self.metric_value = m(cache_gpu[0].cpu().detach().numpy(),cache_gpu[1].cpu().detach().numpy())
                
                elif 'gpu' in m.__name__:
                    self.metric_value = m(cache_gpu[0], cache_gpu[1]).cpu().detach().numpy() #m(cache[:, 0], cache[:, 1])
                else:
                    assert 0,"metirc use gpu or not??"
                self.metrics_dict[m.__name__]=self.metric_value#.cpu().detach().numpy() 
        return self.metrics_dict


class Logg():
    def __init__(self,log_save_path,log_name ='log.txt' ):
        #-----------------------------------------------------------------------------
        mylog = os.path.join(log_save_path , log_name)   
        #display log in console and save log 
        import logging
        self.logger = logging.getLogger(name='training')  # 不加名称设置root logger
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter()
            # '%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')#,
            #'%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
            #datefmt='%Y-%m-%d %H:%M:%S')
            #format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

        # 使用FileHandler输出到文件
        fh = logging.FileHandler(mylog,mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # 添加两个Handler
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

def cal_anomaly_map(fs_list, ft_list, out_size, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size[0], out_size[1]])
    else:
        anomaly_map = np.zeros([out_size[0], out_size[1]])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



# def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
def compute_pro(masks, amaps, num_th= 200) -> None:   
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


# compute F1_score,TPR,FPR,ACC
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.metrics import accuracy_score,f1_score

def compute_metrics(y_true,y_score,threshold):

    acc = accuracy_score(y_true=y_true,y_pred=y_score>threshold)
    f1_score_value = f1_score(y_true=y_true,y_pred=y_score>threshold)

    fpr, tpr, thresholds = roc_curve(y_true, y_score>threshold)
    return f1_score_value,acc,tpr[1],1-fpr[1]


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    
    plt.plot(recs,precs)
    plt.title('PR curve')
    plt.show()
    
    f1s = 2 * precs * recs / (precs + recs+1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr