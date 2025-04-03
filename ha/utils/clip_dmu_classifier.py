import torch
from utils.dmu_embedder import DMUEmbedder
import json
from utils.config_mp import Config
from transformers import CLIPProcessor, CLIPModel


class ClipEmbedder(DMUEmbedder):
    def __init__(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        config_str: str = "configs/clip_pretrain.json",
    ) -> None:
        self.config = Config(config_str)
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # type: ignore
        self.batch_size = self.config.batch_size
        self.anomaly_embeddings = torch.empty(size=(12 * 4, self.config.embedding_dim))
        self.anomaly_addition = self.config.anomaly_addition
        self.anomaly_embedded = False

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        min = torch.min(x.view(x.shape[:-2] + (-1,)), dim=-1, keepdim=True)[0].unsqueeze(-1)
        x = x - min
        m = torch.max(x.view(x.shape[:-2] + (-1,)), dim=-1, keepdim=True)[0].unsqueeze(-1)

        x = x / m

        assert torch.max(x) <= 1, f"Expected max value to be less than 1 but got {torch.max(x)}"
        assert torch.min(x) >= 0, f"Expected min value to be greater than 0 but got {torch.min(x)}"

        x_embedding = torch.Tensor(size=(self.config.clip_length, 3, 224, 224))
        embedding = torch.Tensor(size=(x.size(dim=0), self.config.clip_length, self.config.embedding_dim))

        assert len(x.size()) == 5
        for idx in range(x.size(0)):
            x_embedding = x[idx].to(self.device)
            with torch.no_grad():
                inputs = self.processor(text="",images=x_embedding, return_tensors="pt").to(self.device)
                embedding[idx, :, :] = self.model(**inputs).image_embeds
        return embedding.to(self.device)

    def embedd_text(self, x: list) -> torch.Tensor:
        embeddings = []
        for category_with_text in x:
            text = list(category_with_text.values())[0]
            inputs = self.processor(text=text, images=torch.rand((3,224,224)), return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                e = self.model(**inputs)
                embeddings.append(e.text_embeds)
        embedding = torch.cat(embeddings, dim=0)
        assert len(embedding.size()) == 2
        assert embedding.size(1) == 768
        embedding.to(self.device)
        return torch.cat(embeddings, dim=0)

    def forward_pretrain(self, x: torch.Tensor) -> torch.Tensor:
        if not self.anomaly_embedded:
            with open(self.config.anomaly_description, "r") as f:
                text = json.load(f)
            self.anomaly_embeddings = self.embedd_text(text)
            self.anomaly_embedded = True
        indices = torch.randperm(self.anomaly_embeddings.size(0))[0]
        anomaly_embedding = self.anomaly_embeddings[indices].to(self.device)
        x_embedding = self.forward_eval(x)
        anomaly_embeddings = anomaly_embedding * self.anomaly_addition + x_embedding * (
            1 - self.anomaly_addition
        )
        anomaly_embeddings = anomaly_embeddings.to(self.device)

        return torch.cat([x_embedding, anomaly_embeddings], dim=0).to(self.device)
