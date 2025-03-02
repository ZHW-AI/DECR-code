import os
import torch
import numpy as np
import copy

class best_checksave():
    def __init__(self,verbose=False,save_param=True,save_max=bool):
        self.verbose = verbose
        self.save_param = save_param
        self.save_max = save_max
        self.best_metric = None
        self.update_point = None
        self.best_model_wts    =  None


    def __call__(self,val_metric,model,path,checkpoint_name='model_checkpoint.pth',logger=None):
        self.logger = logger

        if self.best_metric is None:
            self.best_metric = val_metric
            self.update_point = False
        elif val_metric > self.best_metric and self.save_max:
            self.save_checkpoint(val_metric,model,path,checkpoint_name)
            self.update_point = True
            self.best_metric = val_metric
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_metric < self.best_metric and not self.save_max:
            self.save_checkpoint(val_metric,model,path,checkpoint_name)
            self.update_point = True
            self.best_metric = val_metric
            self.best_model_wts = copy.deepcopy(model.state_dict())
        else:
            self.update_point = False
        return self.update_point
            
    def save_checkpoint(self,val_metric,model,path,checkpoint_name):

        if self.save_param:
            torch.save(model.state_dict(), os.path.join(path,checkpoint_name))        
        if self.verbose:
            # print(
            #     f'Validation metric improved ({self.best_metric:.6f} --> {val_metric:.6f}).  Saving model ...')
            self.logger.logger.debug(
                f'Validation check metric improved ({self.best_metric:.6f} --> {val_metric:.6f}).  Saving model ...')
        

        