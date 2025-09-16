# -*- coding: UTF-8 -*-
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torch.cuda.amp import GradScaler, autocast

from models.eeg_encoder import Enc_eeg, Proj_eeg, weights_init_normal
from utils import contrastive_loss, save_checkpoints

def validate_once(val_loader, enc_eeg, proj_eeg, clip_model, criterion_cls):
    enc_eeg.eval(); proj_eeg.eval(); clip_model.eval()
    total_val_loss, batches = 0.0, 0
    device = next(proj_eeg.parameters()).device

    with torch.no_grad(), autocast():
        bar = tqdm(val_loader, desc="Validation", leave=False)
        for batch in bar:
            veeg = batch['eeg'].to(device)
            vimages = batch['image']

            veeg_features = proj_eeg(enc_eeg(veeg))
            vinputs = clip_model.module.processor(images=vimages, return_tensors="pt").to(device)
            vimg_features = clip_model.module.get_image_features(**vinputs)

            veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
            vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

            vloss = contrastive_loss(clip_model.module.logit_scale, veeg_features, vimg_features, criterion_cls)
            total_val_loss += vloss.item()
            batches += 1
            bar.set_postfix(val_loss=f"{vloss.item():.4f}")

    return total_val_loss / max(batches, 1)


def train_and_validate_one_subject(
    subject_id,
    project_dir,
    n_epochs,
    logical_bs,
    micro_bs,
    num_workers,
    result_path,
    model_save_path,
    model_id_prefix,
    seed=None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    from data.things_eeg_dataset import ThingsEEGDataset, collate_fn
    full_train = ThingsEEGDataset(project_dir, subject_id, partition='training')
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_bs, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=logical_bs, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )


    criterion_cls = nn.CrossEntropyLoss().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = nn.DataParallel(CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device))
    clip_model_frozen = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    for p in clip_model_frozen.parameters():
        p.requires_grad = False

    enc_eeg = nn.DataParallel(Enc_eeg(num_channels=63).to(device))
    proj_eeg = nn.DataParallel(Proj_eeg().to(device))


    text_attention_fusion = nn.TransformerEncoderLayer(
        d_model=768, nhead=12, batch_first=True, dim_feedforward=3072
    ).to(device)


    logit_scale = nn.Parameter(torch.tensor(torch.log(torch.tensor(1/0.07)).item(), device=device))
    clip_model.module.processor = clip_processor
    clip_model.module.logit_scale = logit_scale


    eeg_params  = list(enc_eeg.module.parameters()) + list(proj_eeg.module.parameters())
    clip_params = list(clip_model.module.parameters()) + [logit_scale] + list(text_attention_fusion.parameters())
    optim_eeg  = torch.optim.AdamW(eeg_params,  lr=1e-4)
    optim_clip = torch.optim.AdamW(clip_params, lr=1e-6)
    scaler = GradScaler()

    enc_eeg.apply(weights_init_normal)
    proj_eeg.apply(weights_init_normal)

    accumulation_steps = logical_bs // micro_bs
    best_loss_val = float('inf'); best_epoch = 0
    log_f = open(os.path.join(result_path, f"log_subject{subject_id}.txt"), "w")
    model_id = f"{model_id_prefix}_sub{subject_id:02d}"

    for epoch in tqdm(range(n_epochs), desc="Overall Epochs"):
        clip_model.train(); enc_eeg.train(); proj_eeg.train()
        optim_clip.zero_grad(); optim_eeg.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} Training", leave=False)
        for i, batch in enumerate(train_bar):
            eeg_g   = batch['eeg'].to(device)
            images  = batch['image']
            texts_g, texts_c, texts_e = batch['texts']

            with autocast():
                # ===== Finetuned features =====
                proc = clip_processor(text=texts_g, images=images, return_tensors="pt",
                                      padding='max_length', truncation=True).to(device)
                out = clip_model(**proc)
                img_ft_finetuned = out.image_embeds
                txt_g_ft_finetuned = out.text_embeds

                in_c = clip_processor(text=texts_c, return_tensors="pt",
                                      padding='max_length', truncation=True).to(device)
                txt_c_ft_finetuned = clip_model.module.get_text_features(**in_c)

                in_e = clip_processor(text=texts_e, return_tensors="pt",
                                      padding='max_length', truncation=True).to(device)
                txt_e_ft_finetuned = clip_model.module.get_text_features(**in_e)

                txt_seq_finetuned = torch.stack([txt_g_ft_finetuned, txt_c_ft_finetuned, txt_e_ft_finetuned], dim=1)
                fused_seq_finetuned = text_attention_fusion(txt_seq_finetuned)
                txt_fused_finetuned = fused_seq_finetuned[:, 0, :]

                # ===== Frozen features for regularization =====
                with torch.no_grad():
                    frozen_img = clip_model_frozen.get_image_features(pixel_values=proc.pixel_values)
                    frozen_txt_g = clip_model_frozen.get_text_features(
                        **clip_processor(text=texts_g, return_tensors="pt",
                                         padding='max_length', truncation=True).to(device))
                    frozen_txt_c = clip_model_frozen.get_text_features(
                        **clip_processor(text=texts_c, return_tensors="pt",
                                         padding='max_length', truncation=True).to(device))
                    frozen_txt_e = clip_model_frozen.get_text_features(
                        **clip_processor(text=texts_e, return_tensors="pt",
                                         padding='max_length', truncation=True).to(device))
                    txt_seq_frozen = torch.stack([frozen_txt_g, frozen_txt_c, frozen_txt_e], dim=1)
                    fused_seq_frozen = text_attention_fusion(txt_seq_frozen)
                    txt_fused_frozen = fused_seq_frozen[:, 0, :]

                # ===== Regularization =====
                loss_reg_img  = F.mse_loss(img_ft_finetuned, frozen_img.detach())
                loss_reg_text = F.mse_loss(txt_fused_finetuned, txt_fused_frozen.detach())
                loss_reg = loss_reg_img + loss_reg_text

                # ===== Contrastive losses =====
                img_norm = img_ft_finetuned / img_ft_finetuned.norm(dim=-1, keepdim=True)
                txt_norm = txt_fused_finetuned / txt_fused_finetuned.norm(dim=-1, keepdim=True)
                eeg_embeds = proj_eeg(enc_eeg(eeg_g))
                eeg_norm = eeg_embeds / eeg_embeds.norm(dim=-1, keepdim=True)

                loss_img_text = contrastive_loss(logit_scale, img_norm, txt_norm, criterion_cls)
                loss_ei       = contrastive_loss(logit_scale, eeg_norm, img_norm, criterion_cls)
                loss_et       = contrastive_loss(logit_scale, eeg_norm, txt_norm, criterion_cls)

                total_loss = loss_img_text + loss_ei + loss_et +  loss_reg
                total_loss = total_loss / accumulation_steps

            scaler.scale(total_loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optim_clip)
                scaler.step(optim_eeg)
                scaler.update()
                optim_clip.zero_grad(); optim_eeg.zero_grad()

            train_bar.set_postfix(total_loss=f"{(total_loss.item() * accumulation_steps):.4f}")

        # ===== Validation (same epoch) =====
        avg_val_loss = validate_once(val_loader, enc_eeg, proj_eeg, clip_model, criterion_cls)
        if avg_val_loss <= best_loss_val:
            best_loss_val = avg_val_loss
            best_epoch = epoch + 1
            save_checkpoints(model_save_path, model_id, enc_eeg, proj_eeg, clip_model)
            print(f"Epoch {epoch+1}: New best val loss={best_loss_val:.4f}, model saved.")

        log_f.write(f"Epoch {epoch+1}: Avg Val Loss={avg_val_loss:.4f}\n"); log_f.flush()

    log_f.write(f"Best Epoch: {best_epoch}, Best Val Loss: {best_loss_val:.4f}\n")
    log_f.close()
    return best_epoch, best_loss_val
