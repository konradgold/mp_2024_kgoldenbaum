# 1. Load dataset
## 1.1 Binary -> 0: Normal, 1: Anomaly, one video each -> baseline
## 1.2 Binary: 0: Normal, 1: Normal + embedded text (describing anomaly)

# def train(net, normal_loader, abnormal_loader, optimizer, criterion, wind, index):

# 2. Define model
## Image-Bind/Clip/Tokenpacker (?) -> every 2 frames as features
## DMU  -> do we use custom self-attention when using imagebind embeddings?

# 3. Train model 

# 4. Evaluate model

## Fix the damn other eval pipeline