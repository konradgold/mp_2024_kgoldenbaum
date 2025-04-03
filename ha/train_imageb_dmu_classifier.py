from json import load
from math import log
from os import path
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

config = Config("configs/image_bind_train.json")

log = config.log
load_pretrained = False
pretrained_model_path = config.model_checkpoint
num_epochs = config.num_epochs

checkpoint_dir = config.checkpoint_path
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if log:
    wandb.init(project="mcm_ood")
    wandb.config.update({"model": "dmu_classifier"})
    wandb.config.update({"dataset": "ucf_crime"})
    wandb.config.update({"lr": config.lr})
    wandb.config.update({"batch_size": config.batch_size})

model = imagebind_model.ImageBindModel()
embedder = ImageBindEmbedder(model, "configs/image_bind_train.json")
dmu_classifier = DMUClassifier(embedder, "configs/image_bind_train.json")

if load_pretrained:
    dmu_classifier.load_state_dict(torch.load(pretrained_model_path, weights_only=False))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dmu_classifier = dmu_classifier.to(device)

td_normal = TensorDataset(config.data_path_normal)
td_anomaly = TensorDataset(config.data_path_anomaly)

dataloader_normal = DataLoader(td_normal, 
                        batch_size=config.batch_size, shuffle=True, 
                        pin_memory=True, 
                        pin_memory_device="cuda" if torch.cuda.is_available() else "cpu")

dataloader_anomaly = DataLoader(td_anomaly, 
                        batch_size=config.batch_size, shuffle=True, 
                        pin_memory=True, 
                        pin_memory_device="cuda" if torch.cuda.is_available() else "cpu")

dmu_classifier.train()
dmu_classifier.embedding.model.eval()
dmu_classifier.flag = config.mode

criterion = AD_Loss()

optimizer = torch.optim.Adam(dmu_classifier.parameters(), lr = config.lr,
        betas = (0.9, 0.999), weight_decay = 0.00005)

## Train on actual anomaly

normal_dataloader_iterator = iter(dataloader_normal)


print("Training on actual anomaly")
for epoch in range(num_epochs):
    for ainput, alabel in dataloader_anomaly:
        try:
            ninput, nlabel = next(normal_dataloader_iterator)
            ninput = ninput.cuda()
            nlabel = nlabel.cuda()
            ainput = ainput.cuda()
            alabel = alabel.cuda()
            input = torch.cat([ninput, ainput], dim=0)
            _label = torch.cat([nlabel, alabel], dim=0)
            predict = dmu_classifier(input)
            cost, loss = criterion(predict, _label)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            for key in loss.keys():     
                wandb.log({key: loss[key].item()}) if log else None
        except Exception as e:
            print(e)
            continue
    checkpoint_path = os.path.join(checkpoint_dir, f"fine_tune_dmu_classifier_epoch_{epoch + 1}.pth")
    torch.save(dmu_classifier.state_dict(), checkpoint_path)





