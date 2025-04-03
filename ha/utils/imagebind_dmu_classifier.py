import torch
from utils.dmu_embedder import DMUEmbedder
from imagebind.models.imagebind_model import ModalityType
import json
from imagebind import data
from utils.config_mp import Config


class ImageBindEmbedder(DMUEmbedder):
    def __init__(self, model, config_str: str) -> None:
        self.config = Config(config_str)
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = self.config.batch_size
        self.anomaly_embeddings = torch.empty(size=(12*4, self.config.embedding_dim))
        self.anomaly_addition = self.config.anomaly_addition
        self.anomaly_embedded = False
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        x_embedding = torch.Tensor(size=(self.config.clip_length, 3, 224, 224))
        embedding = torch.Tensor(size=(x.size(dim=0), self.config.clip_length, self.config.embedding_dim))
        assert len(x.size()) == 5
        assert x.size(1) == 8
        for idx in range(x.size(0)):
            x_embedding = x[idx].to(self.device)
            with torch.no_grad():
                embedding[idx] = self.model({ModalityType.VISION: x_embedding})["vision"]
        return embedding.to(self.device)
    
    def embedd_text(self, x: list) -> torch.Tensor:
        embeddings = []
        for category_with_text in x:
            text = list(category_with_text.values())[0]
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text, self.device),
            }
            with torch.no_grad():
                embeddings.append(self.model(inputs)["text"])
        embedding = torch.cat(embeddings, dim=0)
        assert len(embedding.size()) == 2
        assert embedding.size(1) == self.config.embedding_dim
        embedding.to(self.device)
        return torch.cat(embeddings, dim=0)

    def forward_pretrain(self, x: torch.Tensor) -> torch.Tensor:
        if not self.anomaly_embedded:
            print("Loading anomaly embeddings")
            with open(self.config.anomaly_description, "r") as f:
                text = json.load(f)
            self.anomaly_embeddings = self.embedd_text(text)
            self.anomaly_embedded = True
        indices = torch.randperm(self.anomaly_embeddings.size(0))[0]
        anomaly_embedding = self.anomaly_embeddings[indices].to(self.device)
        x_embedding = self.forward_eval(x)
        anomaly_embeddings = anomaly_embedding * self.anomaly_addition + x_embedding * (1 - self.anomaly_addition)
        anomaly_embeddings = anomaly_embeddings.to(self.device)

        return torch.cat([x_embedding, anomaly_embeddings], dim=0).to(self.device)

        
        