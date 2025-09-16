# -*- coding: UTF-8 -*-
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class ThingsEEGDataset(Dataset):
    def __init__(self, project_dir, subject_id, partition='training'):
        super().__init__()
        print(f"Loading '{partition}' data for subject {subject_id:02d}...")

        eeg_filename = f'preprocessed_eeg_{partition}.npy'
        eeg_path = os.path.join(project_dir, 'Preprocessed_data_250Hz', f'sub-{subject_id:02d}', eeg_filename)
        self.eeg_data = np.load(eeg_path, allow_pickle=True, mmap_mode='r')['preprocessed_eeg_data']
        print(f"EEG data shape: {self.eeg_data.shape}")

        self.image_files, self.text_files = self._get_aligned_file_paths(project_dir, partition)

        n_eeg = len(self.eeg_data)
        n_img = len(self.image_files)
        n_txt = len(self.text_files['global'])
        if not (n_eeg == n_img == n_txt):
            raise ValueError(f"Data sample counts do not match! EEG: {n_eeg}, Images: {n_img}, Text: {n_txt}")
        print(f"Loading complete. Found and aligned {n_eeg} samples.")

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_repetitions = self.eeg_data[idx]
        eeg_trial_data = np.mean(eeg_repetitions, axis=0)
        eeg_tensor = torch.from_numpy(eeg_trial_data).float().unsqueeze(0)

        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        text_path_g = self.text_files['global'][idx]
        text_path_c = self.text_files['color'][idx]
        text_path_e = self.text_files['emotion'][idx]

        def safe_read(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except FileNotFoundError:
                return ""

        texts = [safe_read(text_path_g), safe_read(text_path_c), safe_read(text_path_e)]
        text_paths = {'global': text_path_g, 'color': text_path_c, 'emotion': text_path_e}

        return {'eeg': eeg_tensor, 'image': image, 'texts': texts, 'label': idx,
                'image_path': image_path, 'text_paths': text_paths}

    def _get_aligned_file_paths(self, project_dir, partition):
        img_part = 'training_images' if partition == 'training' else 'test_images'
        img_root = os.path.join(project_dir, 'Image_set', 'image', img_part)
        concept_folders = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
        image_files = []
        for folder in concept_folders:
            folder_path = os.path.join(img_root, folder)
            image_files.extend(sorted(glob.glob(os.path.join(folder_path, '*.jpg'))))

        num_samples = len(image_files)
        text_files = {'global': [], 'color': [], 'emotion': []}

        text_global_root = os.path.join(project_dir, 'Description', 'deepseek_vl_global1', img_part)
        text_color_root  = os.path.join(project_dir, 'Description', 'deepseek_vl_color1',  img_part)
        text_emotion_root= os.path.join(project_dir, 'Description', 'deepseek_vl_emotion1',img_part)

        num_digits = 7 if partition == 'training' else 5
        for i in range(num_samples):
            base = f"{img_part}_{i + 1:0{num_digits}d}.txt"
            text_files['global'].append(os.path.join(text_global_root,  base))
            text_files['color'].append( os.path.join(text_color_root,   base))
            text_files['emotion'].append(os.path.join(text_emotion_root,base))
        return image_files, text_files


def collate_fn(batch):
    eegs = torch.stack([item['eeg'] for item in batch])
    images = [item['image'] for item in batch]
    texts  = [item['texts'] for item in batch]   # [[g,c,e], ...]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    texts_T = list(map(list, zip(*texts)))       # [3, B]
    image_paths = [item['image_path'] for item in batch]
    text_paths  = [item['text_paths']  for item in batch]
    return {'eeg': eegs, 'image': images, 'texts': texts_T, 'label': labels,
            'image_paths': image_paths, 'text_paths': text_paths}
