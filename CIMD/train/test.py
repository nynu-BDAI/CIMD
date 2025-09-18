# -*- coding: UTF-8 -*-
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

from CIMS.utils import load_checkpoints
from CIMS.EEG_encoder import Enc_eeg, Proj_eeg

@torch.no_grad()
def test_one_subject(
    subject_id,
    project_dir,
    result_path,
    model_save_path,
    model_id_prefix,
    logical_bs,
    num_workers,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = f"{model_id_prefix}_sub{subject_id:02d}"

    from data.things_eeg_dataset import ThingsEEGDataset, collate_fn
    test_set = ThingsEEGDataset(project_dir, subject_id, partition='test')
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=logical_bs, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True
    )

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = nn.DataParallel(CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device))
    enc_eeg = nn.DataParallel(Enc_eeg(num_channels=63).to(device))
    proj_eeg = nn.DataParallel(Proj_eeg().to(device))

    clip_model.module.processor = clip_processor

    load_checkpoints(model_save_path, model_id, enc_eeg, proj_eeg, clip_model)

    enc_eeg.eval(); proj_eeg.eval(); clip_model.eval()
    all_image_embeds, all_labels = [], []

    with autocast():
        bar = tqdm(test_loader, desc="Extracting Test Image Features")
        for batch in bar:
            images, labels = batch['image'], batch['label']
            inputs = clip_model.module.processor(images=images, return_tensors="pt", padding=True).to(device)
            image_embeds = clip_model.module.get_image_features(**inputs)
            all_image_embeds.append(image_embeds)
            all_labels.append(labels)

    gallery = torch.cat(all_image_embeds, dim=0).to(device)
    gallery = gallery / gallery.norm(dim=-1, keepdim=True)
    gallery_labels = torch.cat(all_labels, dim=0)

    # ========= EEG→Image =========
    total = top1 = top3 = top5 = 0
    bar = tqdm(test_loader, desc="EEG→Image Retrieval")
    with autocast():
        for batch in bar:
            eeg, labels = batch['eeg'].to(device), batch['label'].to(device)
            eeg_embeds = proj_eeg(enc_eeg(eeg))
            eeg_embeds = eeg_embeds / eeg_embeds.norm(dim=-1, keepdim=True)
            sim = eeg_embeds @ gallery.t()
            _, top_idx = sim.topk(5, dim=-1)
            retrieved = gallery_labels[top_idx.cpu()]

            total += labels.size(0)
            top1 += (retrieved[:, 0] == labels.cpu()).sum().item()
            top3 += (retrieved[:, :3] == labels.cpu().unsqueeze(1)).sum().item()
            top5 += (retrieved[:, :5] == labels.cpu().unsqueeze(1)).sum().item()

    return top1/total, top3/total, top5/total
