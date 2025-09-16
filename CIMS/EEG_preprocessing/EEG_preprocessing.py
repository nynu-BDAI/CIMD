# -*- coding: UTF-8 -*-
"""
Preprocess the raw EEG data for all subjects in the THINGS-EEG2 dataset.

This script combines functionalities for:
1. Channel selection, epoching, frequency downsampling, and baseline correction.
2. Multivariate noise normalization (MVNN).
3. Sorting data by image conditions and reshaping.
4. Saving the preprocessed data for each subject.

The script loops through all subjects, processing them one by one.
"""
import os
import mne
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.discriminant_analysis import _cov
import scipy
import pickle
import argparse


# =============================================================================
# Utility Functions (Combined from preprocessing_utils.py)
# =============================================================================

def epoching(args, subject_id, data_part, seed):
    """
    This function first converts the EEG data to MNE raw format, and
    performs channel selection, epoching, baseline correction and frequency
    downsampling. Then, it sorts the EEG data of each session according to the
    image conditions.

    Parameters
    ----------
    args : Namespace
       Input arguments.
    subject_id : int
       The ID of the subject to process.
    data_part : str
       'test' or 'training' data partitions.
    seed : int
       Random seed.

    Returns
    -------
    epoched_data : list of float
       Epoched EEG data.
    img_conditions : list of int
       Unique image conditions of the epoched and sorted EEG data.
    ch_names : list of str
       EEG channel names.
    times : float
       EEG time points.
    """
    chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                  'O1', 'Oz', 'O2']

    ### Loop across data collection sessions ###
    epoched_data = []
    img_conditions = []
    for s in range(args.n_ses):
        ### Load the EEG data and convert it to MNE raw format ###
        eeg_dir = os.path.join('Raw_eeg', f'sub-{subject_id:02}',
                               f'ses-{s + 1:02}', f'raw_eeg_{data_part}.npy')
        eeg_path = os.path.join(args.project_dir, eeg_dir)

        if not os.path.exists(eeg_path):
            print(f"Warning: Data file not found for subject {subject_id}, session {s + 1}. Skipping session.")
            continue

        eeg_data = np.load(eeg_path, allow_pickle=True).item()
        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        raw_eeg_data = eeg_data['raw_eeg_data']
        # Convert to MNE raw format
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(raw_eeg_data, info, verbose=False)
        del eeg_data, raw_eeg_data

        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        raw.pick_channels(chan_order, ordered=True)
        # Reject the target trials (event 99999)
        idx_target = np.where(events[:, 2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        ### Epoching, baseline correction and resampling ###
        epochs = mne.Epochs(raw, events, tmin=-.2, tmax=1.0, baseline=(None, 0),
                            preload=True, verbose=False)
        del raw
        # Resampling
        if args.sfreq < 1000:
            epochs.resample(args.sfreq, verbose=False)
        ch_names = epochs.info['ch_names']
        times = epochs.times

        ### Sort the data ###
        data = epochs.get_data()
        events = epochs.events[:, 2]
        img_cond = np.unique(events)
        del epochs
        # Select only a maximum number of EEG repetitions
        max_rep = 20 if data_part == 'test' else 2
        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1],
                                data.shape[2]))
        for i in range(len(img_cond)):
            idx = np.where(events == img_cond[i])[0]
            if len(idx) < max_rep:
                print(
                    f"Warning: Not enough repetitions for condition {img_cond[i]} in subject {subject_id}. Found {len(idx)}, need {max_rep}. Skipping this condition for this session.")
                continue  # Or handle by padding/repeating
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
        del data
        epoched_data.append(sorted_data[:, :, :, 50:])
        img_conditions.append(img_cond)
        del sorted_data

    # Check if any data was loaded
    if not epoched_data:
        raise FileNotFoundError(f"No data processed for subject {subject_id}. Check data paths.")

    ### Output ###
    return epoched_data, img_conditions, ch_names, times


def mvnn(args, epoched_test, epoched_train):
    """
    Compute the covariance matrices of the EEG data and use them to whiten the data.
    """
    ### Loop across data collection sessions ###
    whitened_test = []
    whitened_train = []
    for s in range(args.n_ses):
        session_data = [epoched_test[s], epoched_train[s]]

        ### Compute the covariance matrices ###
        sigma_part = np.empty((len(session_data), session_data[0].shape[2],
                               session_data[0].shape[2]))
        for p in range(sigma_part.shape[0]):
            sigma_cond = np.empty((session_data[p].shape[0],
                                   session_data[0].shape[2], session_data[0].shape[2]))
            print(f"  Calculating covariance for session {s + 1}, partition {p}...")
            for i in tqdm(range(session_data[p].shape[0])):
                cond_data = session_data[p][i]
                if args.mvnn_dim == "time":
                    sigma_cond[i] = np.mean([_cov(cond_data[:, :, t], shrinkage='auto')
                                             for t in range(cond_data.shape[2])], axis=0)
                elif args.mvnn_dim == "epochs":
                    sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]), shrinkage='auto')
                                             for e in range(cond_data.shape[0])], axis=0)
            sigma_part[p] = sigma_cond.mean(axis=0)

        # Use only training data covariance to avoid data leakage
        sigma_tot = sigma_part[1]
        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

        ### Whiten the data ###
        whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
                                                                      session_data[0].shape[2],
                                                                      session_data[0].shape[3])).swapaxes(1, 2)
                                         @ sigma_inv).swapaxes(1, 2), session_data[0].shape))
        whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
                                                                       session_data[1].shape[2],
                                                                       session_data[1].shape[3])).swapaxes(1, 2)
                                          @ sigma_inv).swapaxes(1, 2), session_data[1].shape))

    ### Output ###
    return whitened_test, whitened_train


def save_prepr(args, subject_id, whitened_test, whitened_train, img_conditions_train,
               ch_names, times, seed):
    """
    Merge the EEG data of all sessions together, shuffle repetitions, and save the data.
    """
    ### Merge and save the test data ###
    merged_test = np.concatenate(whitened_test, axis=1)
    del whitened_test
    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:, idx]
    # Insert the data into a dictionary
    test_dict = {
        'preprocessed_eeg_data': merged_test,
        'ch_names': ch_names,
        'times': times
    }
    del merged_test
    # Saving directories
    save_dir = os.path.join(args.project_dir, 'Preprocessed_data_250Hz',
                            f'sub-{subject_id:02}')
    file_name_test = 'preprocessed_eeg_test.npy'
    os.makedirs(save_dir, exist_ok=True)

    print(f"  Saving test data to {os.path.join(save_dir, file_name_test)}")
    with open(os.path.join(save_dir, file_name_test), 'wb') as save_pic:
        pickle.dump(test_dict, save_pic, protocol=4)
    del test_dict

    ### Merge and save the training data ###
    white_data = np.concatenate(whitened_train, axis=0)
    img_cond = np.concatenate(img_conditions_train, axis=0)
    del whitened_train, img_conditions_train
    # Data matrix of shape:
    # Image conditions × EGG repetitions × EEG channels × EEG time points
    unique_img_conds = np.unique(img_cond)
    merged_train = np.zeros((len(unique_img_conds), white_data.shape[1] * args.n_ses,  # Adjusted for multiple sessions
                             white_data.shape[2], white_data.shape[3]))

    for i, cond in enumerate(unique_img_conds):
        idx = np.where(img_cond == cond)[0]
        ordered_data = white_data[idx].reshape(-1, white_data.shape[2], white_data.shape[3])
        merged_train[i, :ordered_data.shape[0]] = ordered_data

    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
    merged_train = merged_train[:, idx]
    # Insert the data into a dictionary
    train_dict = {
        'preprocessed_eeg_data': merged_train,
        'ch_names': ch_names,
        'times': times
    }
    del merged_train

    file_name_train = 'preprocessed_eeg_training.npy'
    print(f"  Saving training data to {os.path.join(save_dir, file_name_train)}")
    with open(os.path.join(save_dir, file_name_train), 'wb') as save_pic:
        pickle.dump(train_dict, save_pic, protocol=4)
    del train_dict


def main():
    # =============================================================================
    # Input arguments
    # =============================================================================
    parser = argparse.ArgumentParser(description="EEG Preprocessing for THINGS-EEG2 dataset")
    # MODIFIED: Changed --sub to --num_subjects to loop through all
    parser.add_argument('--num_subjects', default=10, type=int,
                        help='Total number of subjects to process (e.g., 10 for Things-EEG2).')
    parser.add_argument('--n_ses', default=4, type=int, help='Number of sessions per subject.')
    parser.add_argument('--sfreq', default=250, type=int, help='Downsampling frequency.')
    parser.add_argument('--mvnn_dim', default='epochs', type=str, help="MVNN dimension ('epochs' or 'time').")
    parser.add_argument('--project_dir', default='/mnt/Data/liuxinqi/DATA/THINGS-EEG2/', type=str,
                        help='Root directory of the project.')
    args = parser.parse_args()

    print('>>> EEG data preprocessing <<<')
    print('\nInput arguments:')
    for key, val in vars(args).items():
        print('{:16} {}'.format(key, val))

    # Set random seed for reproducible results
    seed = 20200220

    # =============================================================================
    # Main loop to process all subjects
    # =============================================================================
    for i in range(args.num_subjects):
        subject_id = i + 1
        print(f"\n=========================================================")
        print(f"===== Processing Subject {subject_id:02d}/{args.num_subjects} =====")
        print(f"=========================================================")

        try:
            # 1. Epoch and sort the data
            print("Step 1: Epoching and sorting data...")
            epoched_test, _, ch_names, times = epoching(args, subject_id, 'test', seed)
            epoched_train, img_conditions_train, _, _ = epoching(args, subject_id, 'training', seed)

            # 2. Multivariate Noise Normalization
            print("Step 2: Applying Multivariate Noise Normalization (MVNN)...")
            whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)
            del epoched_test, epoched_train

            # 3. Merge and save the preprocessed data
            print("Step 3: Merging sessions and saving preprocessed data...")
            save_prepr(args, subject_id, whitened_test, whitened_train, img_conditions_train, ch_names,
                       times, seed)

            print(f"----- Subject {subject_id:02d} processed successfully. -----")

        except Exception as e:
            print(f"!!! An error occurred while processing subject {subject_id}: {e}")
            print("!!! Skipping to the next subject.")
            continue

    print("\nAll subjects have been processed.")


if __name__ == '__main__':
    main()