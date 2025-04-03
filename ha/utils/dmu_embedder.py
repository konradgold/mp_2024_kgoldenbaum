import torch
from enum import Enum
from utils.config_mp import Config
from utils.config_enums import Mode


class DMUEmbedder:

    def __init__(self, model, config_str: str) -> None:
        self.config = Config(config_str)
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = self.config.batch_size
        self.anomaly_embeddings = torch.empty(size=(12*4, 768))
        self.anomaly_addition = self.config.anomaly_addition
        self.mode = self.config.mode

    def __call__(self, x) -> torch.Tensor:
        assert len(x.size()) <= 5 and len(x.size()) >= 4
        if len(x.size()) == 4:
            x = x.unsqueeze(0)
        b, t, c, w, h = x.size()
        assert t == self.config.clip_length, f"Expected clip size {self.config.clip_length} but got {t}"
        if self.mode == Mode.TRAIN:
            x = self.forward_train(x)
        if self.mode == Mode.EVAL:
            x = self.forward_eval(x)
        if self.mode == Mode.PRETRAIN:
            x = self.forward_pretrain(x)
            b *= 2
        assert isinstance(x, torch.Tensor)

        assert len(x.size()) == 3
        b_n, t_n, d = x.size()
        assert b_n == b
        assert t_n == t
        x.to(self.device)
        return x
    
    def forward_eval(self, x) -> torch.Tensor:
        return x
    
    def forward_pretrain(self, x) -> torch.Tensor:
        return torch.cat([self.forward_eval(x), self.forward_eval(x)], dim=0)
    
    def forward_train(self, x) -> torch.Tensor:
        return self.forward_eval(x)