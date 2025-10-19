# resnet_feature_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        
    def forward(self, x):
        x = self.layer0(x)
        o1 = self.layer1(x)
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)
        return [o1, o2, o3]

def embedding_concat(feats):
    ups = []
    Hs = [t.shape[-2] for t in feats]
    Ws = [t.shape[-1] for t in feats]
    th, tw = max(Hs), max(Ws)
    for t in feats:
        ups.append(F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False))
    return torch.cat(ups, dim=1)

def random_channel_indices(total_c, m=80, seed=0):
    m = min(m, total_c)
    rng = np.random.default_rng(seed)
    idx = rng.choice(total_c, size=m, replace=False)
    return np.sort(idx)

def compute_gaussian_stats(embeds):
    N,C,H,W = embeds.shape
    embeds = embeds.permute(0,2,3,1).reshape(N, H*W, C)
    means=[]; invs=[]; eps=1e-6
    for l in range(H*W):
        x = embeds[:, l, :].cpu().numpy()
        mu = x.mean(axis=0)
        cov = np.cov(x, rowvar=False) + np.eye(x.shape[1])*eps
        inv = np.linalg.inv(cov)
        means.append(mu); invs.append(inv)
    return np.stack(means,0), np.stack(invs,0), H, W