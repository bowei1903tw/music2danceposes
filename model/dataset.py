import torch
import logging
import numpy as np
import os
import librosa

from natsort import natsorted

import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import Dataset

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

class LoadDataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.audio_dir = config.user_given_waveform
        self.audio_file = os.listdir(self.audio_dir)
        self.npz_dir = config.image_database_npz_dir
        self.beat_resolution = config.beat_resolution
        self.image_file = [['train', natsorted(os.listdir(os.path.join(self.npz_dir, 'train', 'npz')))], 
                           ['val', natsorted(os.listdir(os.path.join(self.npz_dir, 'val', 'npz')))], 
                           ['test', natsorted(os.listdir(os.path.join(self.npz_dir, 'test', 'npz')))]]
        self.image_file_name = natsorted(os.listdir(os.path.join(self.npz_dir, 'train', 'npz'))) + natsorted(os.listdir(os.path.join(self.npz_dir, 'val', 'npz'))) + natsorted(os.listdir(os.path.join(self.npz_dir, 'test', 'npz')))
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(config.Wav2Vec2FeatureExtractor_dir)
        self.resample_rate = self.processor.sampling_rate
        self.audio_segments, self.audio_segments_mask, self.frame_index, self.total_frames = self.load_audio_to_features()
        self.image_database_npz = self.load_image_database()

    def load_image_database(self):
        
        image_database = np.zeros((len(self.image_file_name), 82),dtype=np.float32)
        idx=0
        for type_name, type in self.image_file:
            for _, file_name in enumerate(type):
                image_feature = np.load(os.path.join(self.npz_dir, type_name,'npz',file_name), allow_pickle=True)['poseshape_82']
                image_database[idx] = image_feature
                idx+=1
        
        image_database = torch.from_numpy(image_database)

        return image_database
    
    def convert8th(self, array):
        original_list = list(array)
        result_list = []

        for i in range(len(original_list) - 1):
            result_list.append(original_list[i])
            middle_value = (original_list[i] + original_list[i + 1]) // 2
            result_list.append(middle_value)   
        result_list.append(original_list[-1])
        return np.asarray(result_list)

    def convert16th(self, array):
        original_list = list(array)
        result_list = []

        for i in range(len(original_list) - 1):
            result_list.append(original_list[i])
            
            step = (original_list[i + 1] - original_list[i]) // 4
            result_list.append(original_list[i] + step)
            result_list.append(original_list[i] + 2 * step)
            result_list.append(original_list[i] + 3 * step)   
        result_list.append(original_list[-1])
        return np.asarray(result_list)


    def load_audio_to_features(self):
        assert len(self.audio_file) != 0 ,'Is not given an audio file.'
        assert len(self.audio_file) == 1 ,'Only one audio is allowed.'

        for file_name in self.audio_file:

            full_audio_wave, sampling_rate = librosa.load(os.path.join(self.audio_dir, file_name))

            if self.resample_rate != sampling_rate:
                resampler = T.Resample(sampling_rate, self.resample_rate)
            else:
                resampler = None

            if resampler is None:
                input_audio = full_audio_wave
            else:
                input_audio = resampler(torch.from_numpy(full_audio_wave))

            audio = self.processor(input_audio, sampling_rate=self.resample_rate, return_tensors="np")

            audio = audio['input_values']
            
            sel = ((audio.shape[-1] // 400) * 400)-1
            audio_ = audio[0,:sel]
            total_frames = sel//400
            _, sample_index = librosa.beat.beat_track(y=audio_, sr=self.resample_rate, units='samples', hop_length=400)
            
            _, frame_index = librosa.beat.beat_track(y=audio_, sr=self.resample_rate, units='frames', hop_length=400)
            
            if self.beat_resolution == '4th':
                pass
            elif self.beat_resolution == '8th':
                sample_index = self.convert8th(sample_index)
                frame_index = self.convert8th(frame_index)
            elif self.beat_resolution == '16th':
                sample_index = self.convert16th(sample_index)
                frame_index = self.convert16th(frame_index)

            audio_segments = np.zeros((len(sample_index),48480),dtype=full_audio_wave.dtype)
            audio_segments_mask = np.ones((len(sample_index),48480),dtype=full_audio_wave.dtype)
            
            for segment_idx in range(len(sample_index)):

                if sample_index[segment_idx]-(48480//2) <0 or sel-sample_index[segment_idx] < (48480//2):
                    continue

                seg_audio = audio_[sample_index[segment_idx]-(48480//2):sample_index[segment_idx]+(48480//2)]

                audio_segments[segment_idx] = seg_audio

            audio_segments = torch.from_numpy(audio_segments)
            audio_segments_mask = torch.from_numpy(audio_segments_mask)

        return audio_segments, audio_segments_mask, frame_index, total_frames

