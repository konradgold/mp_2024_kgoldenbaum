import json
import os
from utils.config_enums import Mode


class Config(object):
    def __init__(self, args_path: str|None):
        if args_path is not None and os.path.isfile(args_path):
            with open(args_path, "r") as f:
                args = json.load(f)
        else:
            args = {}
        
        self.root_dir = args.get("root_dir", "mp/mcm_ood/ha/")
        self.mode = Mode(args.get("mode", Mode.EVAL))
        self.lr = args.get("lr", 0.0001)  
        self.embedding_dim = args.get("embedding_dim", 768)
        self.len_feature = 2*self.embedding_dim
        self.model_name = args.get("model_name", "imagebind")
        self.run_name = args.get("run_name", self.mode.value + "_" + self.model_name)
        self.batch_size = args.get("batch_size", 32)
        self.checkpoint_path = args.get("checkpoint_path", f"mp/mcm_ood/ha/checkpoints/{self.run_name}")
        self.data_path_normal = args.get("data_path_normal", 
                                  "/fs00/share/fg-doellner/goldenbaum/cache/image_tensors")
        self.data_path_anomaly = args.get("data_path_anomaly",
                                  "/fs00/share/fg-doellner/goldenbaum/cache/image_tensors")
        self.data_path_test = args.get("data_path_test",
                                  "/fs00/share/fg-doellner/goldenbaum/cache/image_tensors")
        self.num_epochs = args.get("num_epochs", 10)
        self.num_workers = args.get("num_workers", 0)
        self.seed = args.get("seed", 2025)
        self.clip_length = args.get("clip_length", 8)
        self.model_checkpoint = args.get("model_checkpoint", None)
        self.log = args.get("log", True)
        self.anomaly_addition = args.get("anomaly_addition", 0.3)
        self.attention = args.get("attention", False)
        if self.attention == 1:
            self.attention = True
        self.a_nums = args.get("a_nums", 64)
        self.n_nums = args.get("n_nums", 64)
        self.anomaly_description = args.get("anomaly_description", "assets/anomaly_description.json")
  

