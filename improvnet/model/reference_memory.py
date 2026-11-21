"""Independent per-window reference and drum-statistics memory banks."""
import math
import torch
from torch import nn
from improvnet.utils.drum_profile import DRUM_PROFILE_DIM


class ReferenceMemory(nn.Module):
    def __init__(self,feature_dim=256,width=256,tokens=32,heads=4,drum_tokens=8):
        super().__init__()
        self.drum_tokens=drum_tokens
        self.queries=nn.Parameter(torch.randn(tokens,width)/math.sqrt(width))
        self.features=nn.Linear(feature_dim,width,bias=False)
        self.positions=nn.Linear(3,width,bias=False)
        self.profile=nn.Sequential(nn.Linear(DRUM_PROFILE_DIM,width),nn.SiLU(),
                                   nn.Linear(width,width*max(1,drum_tokens)))
        if drum_tokens:
            self.profile_norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,width*2),nn.SiLU(),nn.Linear(width*2,width))
        self.null=nn.Parameter(torch.zeros(1,1,width))

    def forward(self,features,mask,positions,profile):
        batch,windows,frames,width=features.shape
        features=features.reshape(batch*windows,frames,width)
        mask=mask.reshape(batch*windows,frames)
        positions=positions.reshape(batch*windows,frames,3)
        profile=profile.reshape(batch*windows,DRUM_PROFILE_DIM)
        count=batch*windows
        profile_visible=profile[:,:128].gt(0).any(-1,keepdim=True)
        visible=mask.any(-1)
        values=torch.cat((self.null.expand(count,-1,-1),self.features(features)+self.positions(positions)),1)
        valid=torch.cat((torch.ones(count,1,device=mask.device,dtype=torch.bool),mask),1)
        if not self.drum_tokens:
            # V14 checkpoint loading retains its original shared-bank calculation.
            values=torch.cat((values,self.profile(profile).unsqueeze(1)),1)
            valid=torch.cat((valid,profile_visible),1)
            visible=visible|profile_visible[:,0]
        query=self.queries.unsqueeze(0).expand(count,-1,-1)
        memory=query+self.attention(query,values,values,key_padding_mask=~valid,need_weights=False)[0]
        memory=self.norm(memory+self.ffn(memory))*visible[:,None,None]
        if self.drum_tokens:
            # Every slot has its own output weights; statistics never enter the
            # reference resampler. No-drums windows contribute exactly zero.
            drums=self.profile_norm(self.profile(profile).reshape(count,self.drum_tokens,self.queries.size(-1)))
            drums=drums*profile_visible[:,:,None]
            memory=torch.cat((memory,drums),1)
        return memory.reshape(batch,windows*memory.size(1),-1)
