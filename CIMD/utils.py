# -*- coding: UTF-8 -*-
import os
import torch
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def contrastive_loss(logits_scale: torch.nn.Parameter, feat1, feat2, criterion):
    logits = logits_scale.exp() * feat1 @ feat2.t()
    labels = torch.arange(feat1.shape[0], device=feat1.device)
    return (criterion(logits, labels) + criterion(logits.t(), labels)) / 2

def save_checkpoints(save_dir, model_id, enc_eeg, proj_eeg, clip_model):
    torch.save(enc_eeg.module.state_dict(), os.path.join(save_dir, f'{model_id}_Enc_eeg.pth'))
    torch.save(proj_eeg.module.state_dict(), os.path.join(save_dir, f'{model_id}_Proj_eeg.pth'))
    torch.save(clip_model.module.state_dict(), os.path.join(save_dir, f'{model_id}_CLIP.pth'))

def load_checkpoints(load_dir, model_id, enc_eeg, proj_eeg, clip_model):
    enc_eeg.module.load_state_dict(torch.load(os.path.join(load_dir, f'{model_id}_Enc_eeg.pth')))
    proj_eeg.module.load_state_dict(torch.load(os.path.join(load_dir, f'{model_id}_Proj_eeg.pth')))
    clip_model.module.load_state_dict(torch.load(os.path.join(load_dir, f'{model_id}_CLIP.pth')))
