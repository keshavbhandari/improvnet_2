"""Compress frozen global/local reference features and drum controls into 32 tokens."""
import math
import torch
from torch import nn
from improvnet.utils.drum_profile import DRUM_PROFILE_DIM


class ReferenceMemory(nn.Module):
    def __init__(self,feature_dim=256,width=256,tokens=32,heads=4):
        super().__init__()
        self.queries=nn.Parameter(torch.randn(tokens,width)/math.sqrt(width))
        self.features=nn.Linear(feature_dim,width,bias=False)
        self.positions=nn.Linear(3,width,bias=False)
        self.profile=nn.Sequential(nn.Linear(DRUM_PROFILE_DIM,width),nn.SiLU(),nn.Linear(width,width))
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,width*2),nn.SiLU(),nn.Linear(width*2,width))
        self.null=nn.Parameter(torch.zeros(1,1,width))

    def forward(self,features,mask,positions,profile):
        batch=features.size(0)
        visible=mask.any(-1)|profile[:,:128].gt(0).any(-1)
        values=self.features(features)+self.positions(positions)
        profile_visible=profile[:,:128].gt(0).any(-1,keepdim=True)
        values=torch.cat((self.null.expand(batch,-1,-1),values,self.profile(profile).unsqueeze(1)),1)
        valid=torch.cat((torch.ones(batch,1,device=mask.device,dtype=torch.bool),mask,profile_visible),1)
        query=self.queries.unsqueeze(0).expand(batch,-1,-1)
        memory=query+self.attention(query,values,values,key_padding_mask=~valid,need_weights=False)[0]
        memory=self.norm(memory+self.ffn(memory))
        return memory*visible[:,None,None]
