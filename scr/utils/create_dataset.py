import os
import sys
sys.path.append('.')
import warnings

warnings.filterwarnings("ignore", message=".*had to be resampled from.*")
warnings.filterwarnings("ignore", message="Warning: input samples dtype is np.float64. Converting to np.float32")
warnings.filterwarnings("ignore", message="Xing stream size off by more than 1%")
warnings.filterwarnings("default", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import pandas as pd
import librosa as lb

from torch.utils.data import Dataset

from scr.utils.base_utils import (crop_or_pad,
                        compute_melspec,
                        mono_to_color,
                        normalize)


class BirdDataset(Dataset):

    def __init__(self,
                 df,
                 path_to_folder_with_audio,
                 is_train=False,
                 # Частота дискретизации и продолжительность в сек
                 length=32000 * 5,
                 sr=32000,
                 n_mels=256,
                 fmin=200,
                 fmax=16000,
                 n_fft=1024,
                 hop_length=256,
                 ):

        self.df = df
        self.path_to_folder_with_audio = path_to_folder_with_audio
        self.is_train = is_train
        self.length = length
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # Создаем path до файла
        if row['version'] == 'new':
            path_to_file = os.path.join(self.path_to_folder_with_audio, "add_audio", row['filename'])
        else:
            path_to_file = os.path.join(self.path_to_folder_with_audio, "base_audio", row['filename'])

        # Считываем аудиозапись
        samples, sample_rate = lb.load(path_to_file,
                                       sr=32000,
                                       dtype="float32")

        # Вырезаем случайный кусочек или добавляем паддинги до нашей длины
        crop_audio = crop_or_pad(samples, length=self.length)

        # Тут нужно будет добавить аугментации в будущем
        sample = crop_audio

        # Создаем мелграмму
        melspec = compute_melspec(sample,
                                  sr=self.sr,
                                  n_mels=self.n_mels,
                                  fmin=self.fmin,
                                  fmax=self.fmax,
                                  n_fft=self.n_fft,
                                  hop_length=self.hop_length)

        # Создаем из аудиозаписи в 1 канал RGB изображение и нормализуем его
        image = mono_to_color(melspec)
        image = normalize(image)
        image = torch.tensor(image).float()

        # Создаю тензор с лейблом, для y вектора
        target_value = np.array([0] * 264, dtype=float)
        target_value[row['label']] = 1

        return image, target_value


if __name__ == "__main__":
    df = pd.read_csv("../../data/data.csv")
    data = BirdDataset(df=df,
                       path_to_folder_with_audio='../../data')

    test_sample = data[123]
    print('Test_sample created')
    print('Finish')
