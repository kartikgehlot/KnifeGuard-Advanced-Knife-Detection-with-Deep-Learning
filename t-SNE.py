import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
from config import config
import gc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')


feature_extractor = timm.create_model(config.model_name, features_only=True, out_indices=[4])
feature_extractor.load_state_dict(torch.load(config.checkpoints),strict = False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)
val_imlist = pd.read_csv("test.csv")
val_gen = knifeDataset(val_imlist,mode="val")
for i, (images, target, fnames) in enumerate(val_gen):
    img = images.cuda(non_blocking=True)
    if i == 0:
        features = feature_extractor(img.unsqueeze(0))[0]
        label = np.expand_dims(target,axis=0)
    else:
        features = torch.cat([features,feature_extractor(img.unsqueeze(0))[0]],0)
        label = np.concatenate([label,np.expand_dims(target,axis=0)],0)
    img=img.to('cpu')
    img=None
    gc.collect()
    if i == 90:
        break
features=features.cpu().detach().view(features.size(0), -1).numpy()
tsne = TSNE(n_components=2,perplexity=15, random_state=42)
embedded_features = tsne.fit_transform(features)

# Plot the embedded features
plt.figure(figsize=(8, 6))
plt.scatter(embedded_features[:, 0], embedded_features[:, 1],c=label, cmap='viridis')
plt.show()
 # for feat in features:
 #    plt.imshow(feat[0].transpose(0,2).sum(-1).cpu().detach().numpy())
 #    # plt.imshow(feat[0][0].cpu().detach().numpy())
 #    plt.show()

 