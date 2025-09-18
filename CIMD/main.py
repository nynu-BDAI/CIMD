# -*- coding: UTF-8 -*-
import os
import argparse
import datetime
import gc
import numpy as np
import pandas as pd
import torch

from utils import set_seed
from train.train_val import train_and_validate_one_subject
from train.test import test_one_subject

def parse_args():
    parser = argparse.ArgumentParser(description='CIMD: THINGS-EEG2')
    parser.add_argument('--project_dir', default='/mnt/DATA/THINGS-EEG2/', type=str)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--num_sub', default=10, type=int)
    parser.add_argument('--logical_batch_size', default=64, type=int,
                        help='Effective batch size after gradient accumulation.')
    parser.add_argument('--micro_batch_size', default=16, type=int,
                        help='Actual per-GPU batch size.')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=2025, type=int)

    # 路径/GPU
    parser.add_argument('--result_path', default='/mnt/Data/result/', type=str)
    parser.add_argument('--gpus', default='2', type=str, help='Comma-separated GPU ids, e.g., "0" or "0,1"')
    return parser.parse_args()

def main():
    args = parse_args()

    # ====== GPU ======
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    RESULT_PATH = args.result_path
    MODEL_SAVE_PATH = os.path.join(RESULT_PATH, 'model_base/')
    MODEL_ID_PREFIX = 'CIMS'
    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    print(f"Script started at: {datetime.datetime.now()}")

    for i in range(args.num_sub):
        subject_id = i + 1
        print(f"\n===== Train+Val for subject {subject_id} =====")
        seed_n = np.random.randint(args.seed)
        set_seed(seed_n)

        best_epoch, best_val_loss = train_and_validate_one_subject(
            subject_id=subject_id,
            project_dir=args.project_dir,
            n_epochs=args.epoch,
            logical_bs=args.logical_batch_size,
            micro_bs=args.micro_batch_size,
            num_workers=args.num_workers,
            result_path=RESULT_PATH,
            model_save_path=MODEL_SAVE_PATH,
            model_id_prefix=MODEL_ID_PREFIX,
            seed=seed_n
        )
        print(f"[Subject {subject_id}] Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
        gc.collect(); torch.cuda.empty_cache()

    results = []
    for i in range(args.num_sub):
        subject_id = i + 1
        print(f"\n===== Test for subject {subject_id} =====")
        top1, top3, top5 = test_one_subject(
            subject_id=subject_id,
            project_dir=args.project_dir,
            result_path=RESULT_PATH,
            model_save_path=MODEL_SAVE_PATH,
            model_id_prefix=MODEL_ID_PREFIX,
            logical_bs=args.logical_batch_size,
            num_workers=args.num_workers,
        )
        results.append([top1, top3, top5])
        gc.collect(); torch.cuda.empty_cache()

    results_df = pd.DataFrame(results, columns=['Top1', 'Top3', 'Top5'],
                              index=[f'Sub{i + 1}' for i in range(args.num_sub)])
    results_df.loc['Average'] = results_df.mean()
    print("\n===== Average results for all subjects =====")
    print(results_df)
    results_df.to_csv(os.path.join(RESULT_PATH, 'summary_results.csv'))

    print(f"Script finished at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
