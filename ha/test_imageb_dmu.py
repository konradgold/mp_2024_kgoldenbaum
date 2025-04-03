from logging import config
from math import log
import wandb
from utils.dmu_classifier import DMUClassifier
from utils.imagebind_dmu_classifier import ImageBindEmbedder
from imagebind.models import imagebind_model
from utils.config_enums import Mode
from utils.kl_train_data_loader import TensorDataset
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from utils.config_mp import Config

config_path = "configs/image_bind_test.json"
config = Config(config_path)

log = config.log
num_epochs = config.num_epochs
pretrained_model_path = config.model_checkpoint

checkpoint_dir = config.checkpoint_path
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if log:
    wandb.init(project="mcm_ood")
    wandb.config.update({"model": config_path})
    wandb.config.update({"dataset": "shanghai"})
    wandb.config.update({"lr": config.lr})
    wandb.config.update({"batch_size": config.batch_size})

model = imagebind_model.ImageBindModel()
embedder = ImageBindEmbedder(model, config_path)
dmu_classifier = DMUClassifier(embedder, config_path)

if pretrained_model_path:
    dmu_classifier.load_state_dict(torch.load(pretrained_model_path, weights_only=False))
else:
    raise ValueError("No pretrained model found")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dmu_classifier = dmu_classifier.to(device)
td = TensorDataset(config.data_path_test)
dataloader = DataLoader(td, 
                        batch_size=config.batch_size, shuffle=True, 
                        pin_memory=True, 
                        pin_memory_device="cuda" if torch.cuda.is_available() else "cpu")
dmu_classifier.eval()
dmu_classifier.flag = config.mode

frame_predict = None
        
cls_label = []
cls_pre = []
temp_predict = torch.zeros((0)).cuda()

frame_gt = None

with torch.no_grad():
    for i in range(len( dataloader)):
        input, label = next(iter(dataloader))
        input = input.cuda()
        label = label.cuda()
        frame_gt = torch.cat([frame_gt, label.flatten()], dim=0) if frame_gt is not None else label.flatten()
        res = dmu_classifier(input)
        a_predict = res["frame"]
        if i == 0:
            print("a_predict", a_predict)
        temp_predict = torch.cat([temp_predict, a_predict], dim=0)
        if (i + 10) % 1 == 0 :
            a_predict = torch.sigmoid(temp_predict).flatten().cpu().numpy()
            cls_pre += [1 if a_p>0.5 else 0 for a_p in a_predict]
            cls_label.append(label.flatten().int().cpu().numpy())  
            fpre_ = a_predict
            if frame_predict is None:         
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])
            
            temp_predict = torch.zeros((0)).cuda()
   
    print("frame_predict", frame_predict)
    print("frame_gt", frame_gt)
    frame_gt = frame_gt.cpu().numpy() # type: ignore
    fpr,tpr,_ = roc_curve(frame_gt, frame_predict) # type: ignore
    auc_score = auc(fpr, tpr)
    
    corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
    accuracy = corrent_num / (len(cls_pre))
        
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict,) #type: ignore
    ap_score = auc(recall, precision)
    if log:
        wandb.log({"roc_auc": auc_score})
        wandb.log({"accuracy": accuracy})
        wandb.log({"pr_auc": ap_score})
        wandb.log({"scores": frame_predict})
        wandb.log({"roc_curve": (tpr,fpr)})
        wandb.log({"auc": auc_score})
        wandb.log({"ap": ap_score})
        wandb.log({"ac": accuracy})

