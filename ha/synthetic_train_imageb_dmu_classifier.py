from math import log
from os import path
import time
import wandb
from utils.dmu_classifier import DMUClassifier
from utils.imagebind_dmu_classifier import ImageBindEmbedder
from imagebind.models import imagebind_model
from utils.kl_train_data_loader import TensorDataset
import torch
from torch.utils.data import DataLoader
from utils.dmu.train import AD_Loss
import os
from utils.config_mp import Config

config_path = "configs/image_bind_pretrain.json"

config = Config(config_path)

log = config.log
num_epochs = config.num_epochs

checkpoint_dir = config.checkpoint_path

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if log:
    wandb.init(project="mcm_ood")
    wandb.config.update({"model": config_path})
    wandb.config.update({"dataset": "ucf"})
    wandb.config.update({"lr": config.lr})
    wandb.config.update({"batch_size": config.batch_size})


model = imagebind_model.ImageBindModel()
embedder = ImageBindEmbedder(model, config_path)
dmu_classifier = DMUClassifier(embedder, config_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dmu_classifier = dmu_classifier.to(device)
td = TensorDataset(config.data_path_normal)
dataloader = DataLoader(td, 
                        batch_size=config.batch_size, shuffle=True, 
                        pin_memory=True, 
                        pin_memory_device="cuda" if torch.cuda.is_available() else "cpu")
dmu_classifier.train()
dmu_classifier.flag = config.mode

criterion = AD_Loss()

optimizer = torch.optim.Adam(dmu_classifier.parameters(), lr = config.lr,
        betas = (0.9, 0.999), weight_decay = 0.00005)

## Pretrain
start = time.time()
for epoch in range(num_epochs):
    for ninput, nlabel in dataloader:
        try:
            ninput = ninput.cuda()
            nlabel = nlabel.cuda()
            alabel = torch.ones_like(nlabel)
            _label = torch.cat([nlabel, alabel], dim=0)
            predict = dmu_classifier(ninput)
            cost, loss = criterion(predict, _label)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            for key in loss.keys():     
                wandb.log({key: loss[key].item()}) if log else None
        except Exception as e:
            print(e)
            continue
        if time.time() - start > 36000:
            checkpoint_path = os.path.join(checkpoint_dir, f"dmu_ib_ucf_epoch_{time.time()}.pth")
            start = time.time()
            torch.save(dmu_classifier.state_dict(), checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_dir, f"dmu_ib_ucf_epoch_{epoch + 1}.pth")
    torch.save(dmu_classifier.state_dict(), checkpoint_path)





