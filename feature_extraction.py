import glob
import pandas as pd
import numpy as np
import mne
import hfda
import neurokit2 as nk
import re

folder_path_training_set = '/home/zvone/faks/Zavrsni\\ rad/Dataset/Training\\ set/SET+events'
file_pattern_training_set = '*.set'
file_paths_training_data = glob.glob(folder_path_training_set.replace('\\', '') + '/' + file_pattern_training_set)

folder_path_test_set = '/home/zvone/faks/Zavrsni\\ rad/Dataset/Test\\ set/SET+events'
file_pattern_test_set = '*.set'
file_paths_test_data = glob.glob(folder_path_test_set.replace('\\', '') + '/' + file_pattern_test_set)

def filter_brainwaves(raw, freq_range):
    filtered_raw = raw.copy().filter(l_freq=freq_range[0], h_freq=freq_range[1])
    return filtered_raw

def load_and_preprocess_eeg_file(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw.crop(tmax=400)
    low_freq, high_freq = 1.0, 40.0
    raw.filter(low_freq, high_freq)
    n_components = min(20, len(raw.ch_names))
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
    ica.fit(raw)
    blink_indices, _ = ica.find_bads_eog(raw, ch_name='Fp1')
    ica.exclude = blink_indices
    raw_clean = ica.apply(raw.copy())

    brainwave_ranges = {'Delta': [0.5, 4], 'Theta': [4, 8], 'Alpha': [8, 12], 'Beta': [12, 30], 'Gamma': [30, 45]}
    brainwaves = {}
    for wave, freq_range in brainwave_ranges.items():
        brainwaves[wave] = filter_brainwaves(raw_clean, freq_range)
    return brainwaves

def calculate_fractal_dimensions(signal):
    return hfda.measure(signal, k_max=10), nk.fractal_sevcik(signal)[0], nk.fractal_katz(signal)[0], nk.fractal_petrosian(signal)[0]

def extract_features_and_load_into_df(file_paths):
    data = []
    for file_path in file_paths:
        file_name = file_path.split('/')[-1]
        brainwaves = load_and_preprocess_eeg_file(file_path)
        for wave, raw in brainwaves.items():
            signal = raw.get_data()
            for i, ch_signal in enumerate(signal):
                hfd, sfd, kfd, pfd = calculate_fractal_dimensions(ch_signal)

                id_match = re.search(r'EEG_(\d+)\.set', file_name)
                if id_match:
                    id_num = id_match.group(1)
                else:
                    id_num = '6673'

                features = {
                    'id': id_num,
                    'file_name': file_name,
                    f'higuchi_{wave}_ch{i}': hfd,
                    f'sevcik_{wave}_ch{i}': sfd,
                    f'katz_{wave}_ch{i}': kfd,
                    f'petrosian_{wave}_ch{i}': pfd
                }
                data.append(features)
    df = pd.DataFrame(data)
    return df

def read_depressed_value(df, training):
    if training:
        sheet = 'Training_for_input'
    else:
        sheet = 'Test_for_input'

    # Read "Training" or "Test" sheet from subject_info.xlsx file
    info_df = pd.read_excel('/home/zvone/faks/Zavrsni rad/Dataset/subject_info_changed.ods', sheet_name=sheet)

    info_df['id'] = info_df['id'].astype('str')
    df['id'] = df['id'].astype('str')

    # Merge info_df with your existing DataFrame on "id" column
    merged_df = pd.merge(df, info_df[['id', 'diagnosis']], on='id')

    def is_depressive(row):
        if row['diagnosis'] == 'healthy':
            return 0
        else:
            return 1

    merged_df['is_depressive'] = merged_df.apply(is_depressive, axis=1)

    final_df = merged_df.drop(columns='diagnosis')
    
    return final_df


df_training_set = extract_features_and_load_into_df(file_paths_training_data)
df_training_set = df_training_set.groupby(['id', 'file_name']).first().reset_index()

df_final_training_set = read_depressed_value(df_training_set, True)
df_final_training_set.to_csv('Training_set.csv', index=False)

df_test_set = extract_features_and_load_into_df(file_paths_test_data)
df_test_set = df_test_set.groupby(['id', 'file_name']).first().reset_index()

df_final_test_set = read_depressed_value(df_test_set, False)
df_final_test_set.to_csv('Test_set.csv', index=False)