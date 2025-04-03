from functools import cache
from utils.pipeline import Pipeline, CollectionEvaluator
from utils.video_loader import VideoLoaderUCF
from utils.embedder_pipe_module import ClipEmbedder
import wandb
from torch.utils.data import DataLoader
from classifier.nn_definition import EmbeddingDataset, PlainNNClassifier, StratifiedSampler, DataModule
from utils.cluster_analyzer import ClusterAnalyzer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
import os
import torch

video_path = os.environ.get("VIDEO_PATH", "/mnt/ssd2/UCF_Crimes/Videos/")
annotation_path_train = os.environ.get("ANNOTATION_PATH_TRAIN", "/mnt/ssd2/UCF_Crimes/annotations/train.json")
annotation_path_test = os.environ.get("ANNOTATION_PATH_TEST", "/mnt/ssd2/UCF_Crimes/annotations/test.json")
cache_path = os.environ.get("CACHE_PATH", "/hpi/fs00/home/konrad.goldenbaum/emb_cache/")
ckpt_path = os.environ.get("CKPT_PATH", "/home/mp/konrad/nn_results/cpt")
log_path = os.environ.get("LOG_PATH", "/home/mp/konrad/nn_results/lightning_logs")
wandb_api = os.environ.get("wandb_api=", "")

print(video_path)
print(annotation_path_train)

#### 1. Clip ####

wandb.login(key=wandb_api)
wandb.init(project="mp2024-mcm")
wandb.config.update({"model": "clip", "dataset": "ucf_crimes", "method": "forest"})
if wandb.run is not None:   
    wandb.run.name = "clip_ucf_crimes_forest"


embedder = ClipEmbedder()
video_loader = VideoLoaderUCF(annotation_path_train, video_path, split="train", cache_path=cache_path)
test_video_loader = VideoLoaderUCF(annotation_path_test, video_path, split="test", cache_path=cache_path)

### 1.1 Gaussian Mixture ###

cluster_analyzer = ClusterAnalyzer(n_components = 15, warm_start=True)
print(video_loader[0])
pipeline = Pipeline(video_loader, embedder, cluster_analyzer, test_video_loader=test_video_loader, evaluator=CollectionEvaluator())
result_pipe = pipeline.run()

wandb.log(result_pipe)


### 1.2 NN ###

train_dataset = EmbeddingDataset(cache_path, annotation_path_train)
test_dataset = EmbeddingDataset(cache_path, annotation_path_test)
batch_size = 32

classifier = PlainNNClassifier()
ts = StratifiedSampler(train_dataset, batch_size) 
train_loader = DataLoader(
   train_dataset, shuffle=False, num_workers=8, batch_sampler=ts
)
test_loader = DataLoader(
   test_dataset, shuffle=False, num_workers=8, batch_sampler=StratifiedSampler(test_dataset, batch_size)

)

datamodel = DataModule(train_dataset, test_dataset, 32)

checkpoint_callback = ModelCheckpoint(
   dirpath=ckpt_path,
   monitor="val_loss",
   filename="emb_classifier-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
   save_top_k=3,
   mode="min",
)

logger = TensorBoardLogger(save_dir=log_path, name="emb_classifier")
early_stopping = EarlyStopping(monitor="val_fbeta", patience=20, mode="max", verbose=False)

trainer = L.Trainer(
   max_epochs=50,
   callbacks=[checkpoint_callback, early_stopping],
   logger=logger,
   accelerator="gpu" if torch.cuda.is_available() else "cpu",
   devices="auto",
)

trainer.fit(classifier, datamodel)

