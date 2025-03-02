import os
import copy 
from visdom import Visdom
from tensorboardX import SummaryWriter
import torch


class vis_classifi(object):
    def __init__(self,visualization,env,metrics,visdom_port,model_save_path):
        self.visualization = visualization
        self.env = env
        self.metrics = metrics

        if 'visdom' in visualization:
            self.viz = Visdom(env=env, port = visdom_port)
            self.viz.line([[0.],[0.]],[0.],win='train',opts=dict(title='train',legend=['train_loss','val_loss']))
            for m in metrics:
                self.viz.line([[0.],[0.]],[0.],win='test_'+m.__name__,opts=dict(title='test_'+m.__name__,legend=["train_"+m.__name__,"val_"+m.__name__]))

        # tensorboardX
        if 'tensorboardX' in visualization:
            self.writer = SummaryWriter(os.path.join(model_save_path,'runs',env))

    def __call__(self,train_log,val_log):
        train_img , train_label,train_pred = train_log['inputs'],train_log['labels']  ,train_log['preds']
        val_img   , val_label  ,val_pred   = val_log['inputs']  ,val_log['labels']    ,val_log['preds'] 
        train_metrics_dict = train_log["metrics_value"]  
        val_metrics_dict = val_log["metrics_value"]
        train_epoch_loss = train_log['train_epoch_avg_loss']
        val_epoch_loss = val_log['val_epoch_avg_loss']
        epoch_num = train_log['epoch']

        if 'visdom' in self.visualization:
            #vis train
            self.viz.images(train_img[0:,[2,1,0],:,:],win='train_img',opts=dict(title='train_imgs'))
            # self.viz.images(train_label[:,:,:,:],win='train_label',opts=dict(title='train_labels'))
            # self.viz.images(train_pred[:,:,:,:],win='train_preds',opts=dict(title='train_preds'))
            #vis val
            self.viz.images(val_img[0:,[2,1,0],:,:],win='val_img',opts=dict(title='val_imgs'))
            # self.viz.images(val_label[:,:,:,:],win='val_label',opts=dict(title='val_labels'))
            # self.viz.images(val_pred[:,:,:,:],win='val_preds',opts=dict(title='val_preds')) 
            """
            #vis train :img+label
            train_img1  = copy.deepcopy(train_img)
            train_img2  = copy.deepcopy(train_img)
            train_img1[:,[0],:,:] = train_img1[:,[0],:,:].add(train_label)
            train_img2[:,[0],:,:] = train_img2[:,[0],:,:].add(train_pred)
            train_img1 = torch.clip(train_img1,min=0,max=1)
            train_img2 = torch.clip(train_img2,min=0,max=1)         
            vis_imgs_train =torch.cat([train_img,train_img1,train_img2],dim=2)
            self.viz.images(vis_imgs_train[: ,[2,1,0],:,:],win='train_labels_preds',opts=dict(title='train_labels_preds'))
            #vis val:img+label
            val_img1  = copy.deepcopy(val_img)
            val_img2  = copy.deepcopy(val_img)
            val_img1[:,[0],:,:] = val_img1[:,[0],:,:].add(val_label)
            val_img2[:,[0],:,:] = val_img2[:,[0],:,:].add(val_pred)
            val_img1 = torch.clip(val_img1,min=0,max=1)
            val_img2 = torch.clip(val_img2,min=0,max=1)         
            vis_imgs_val =torch.cat([val_img,val_img1,val_img2],dim=2)
            self.viz.images(vis_imgs_val[: ,[2,1,0],:,:],win='val_labels_preds',opts=dict(title='val_labels_preds'))           
            """
            #loss line
            self.viz.line([[train_epoch_loss],[val_epoch_loss]],[epoch_num],win='train',update='append')
            for m in self.metrics:
                self.viz.line([[train_metrics_dict[m.__name__]],[val_metrics_dict[m.__name__]]],[epoch_num],\
                                            win='test_'+m.__name__,update='append')
        if 'tensorboardX' in self.visualization:

            for m in self.metrics:
                self.writer.add_scalars(m.__name__,{'train_'+ m.__name__: train_metrics_dict[m.__name__],'test_'+ m.__name__: val_metrics_dict[m.__name__]}, \
                                                                        global_step=epoch_num)
                                                                    
            self.writer.add_scalar('train_loss', train_epoch_loss, global_step=epoch_num)
            self.writer.add_scalars("loss", {'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss},global_step= epoch_num)
        
    

