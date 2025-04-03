from typing import Any
from utils.dmu.model import WSAD, ADCLS_head
from torch import nn
from utils.dmu.memory import Memory_Unit
from utils.dmu.translayer import Transformer
import torch
from utils.dmu_embedder import DMUEmbedder, Mode
from utils.config_mp import Config



class DMUClassifier(WSAD):
    def __init__(self, embedder: DMUEmbedder, config_path: str) :
        '''
        Args:
            embedder: nn.Module
            attention: bool, indicates whether to use (and potentially train) attention
        '''
        self.config = Config(config_path)
        torch.nn.Module.__init__(self)

        self.flag = self.config.mode
        self.n_nums = self.config.n_nums
        self.clip_size = self.config.clip_length

        
        self.embedding = embedder
        self.embedding.mode = self.config.mode

        for param in self.embedding.model.parameters():
            param.requires_grad = False

        self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ADCLS_head(2*self.config.embedding_dim, 1)
        self.Amemory = Memory_Unit(nums=self.config.a_nums, dim=self.config.embedding_dim)
        self.Nmemory = Memory_Unit(nums=self.config.n_nums, dim=self.config.embedding_dim)
        if self.config.attention:
            self.selfatt = Transformer(self.config.embedding_dim, 2, 4, 128, self.config.embedding_dim, dropout = 0.5)
        else:
            self.selfatt = lambda x: x
        self.encoder_mu = nn.Sequential(nn.Linear(self.config.embedding_dim, self.config.embedding_dim))
        self.encoder_var = nn.Sequential(nn.Linear(self.config.embedding_dim, self.config.embedding_dim))

        self.relu = nn.ReLU()

