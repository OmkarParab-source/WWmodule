import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from sonopy import mfcc_spec

class MFCC(nn.Module):

    def __init__(
        self,
        sample_rate,
        fft_size=400,
        window_stride=(400,200),
        num_filt=40,
        num_coeffs=40
    ):
        super(MFCC, self).__init__()
        self.sample_rate=sample_rate
        self.window_stride=window_stride
        self.fft_size=fft_size
        self.num_filt=num_filt
        self.num_coeffs=num_coeffs
        self.mfcc = lambda x:mfcc_spec(
            x,
            self.sample_rate,
            self.window_stride,
            self.fft_size,
            self.num_filt,
            self.num_coeffs
        )
    
    def forward(self, x):
        return torch.Tensor(self.mfcc(x.squeeze(0).numpy())).transpose(0,1).unsqueeze(0)


def get_mfcc(sample_rate):
    return MFCC(sample_rate=sample_rate)

class SpecAugment(nn.Module):

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        
        super(SpecAugment, self).__init__()
        self.rate=rate
        
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, data_csv, sample_rate=16000, test=False):
        self.sr=sample_rate
        self.data=pd.read_csv(data_csv)
        if test:
            self.audio_transform=get_mfcc(sample_rate)
        else:
            self.audio_transform = nn.Sequential(
                get_mfcc(sample_rate),
                SpecAugment(rate=0.5)
            )
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        try:
            file_path=self.data.path.iloc[idx]
            waveform, sr = torchaudio.load(file_path, normalize=False)
            mfcc=self.audio_transform(waveform)
            label=self.data.label.iloc[idx]
        except Exception as e:
            print(str(e), file_path)
            return self.__getitem__(torch.randint(0, len(self), (1,)))
        # print(file_path, waveform.shape)
        return mfcc, label


def combine_data(data):
    mfccs=[]
    labels=[]
    for d in data:
        mfcc, label=d
        mfccs.append(mfcc.squeeze(0).transpose(0,1))
        labels.append(label)
    
    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    mfccs = mfccs.transpose(0, 1)
    labels = torch.Tensor(labels)
    return mfccs, labels