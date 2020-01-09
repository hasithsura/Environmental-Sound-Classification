import librosa
import numpy as np
import os
from fastai.vision import PreProcessor
import tqdm
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def spectrogram(file_path,params):
    wav,sr = librosa.load(file_path,sr=params.sr)
    if wav.shape[0]<5*params.sr:
      wav=np.pad(wav,int(np.ceil(5*params.sr//wav.shape[0])),mode='reflect')
    wav=wav[:5*params.sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=params.n_fft,
                hop_length=params.hop_length,n_mels=params.n_mels,fmin=params.fmin,fmax=params.fmax)
    specdb=librosa.power_to_db(spec,top_db=params.top_db)
    specdb=specdb.astype(np.float32)
    return np.rollaxis(mono_to_color(specdb),2,0)

class AudioPreProcessor(PreProcessor):
    def __init__(self,base_path,params,**kwargs):
        super().__init__(**kwargs)
        self.base_path=base_path
        self.params=params
    def process_one(self, item):
        return spectrogram(os.path.join(self.base_path,item),self.params).astype(np.double)
    def process(self, ds):
        ds.items = np.array([self.process_one(item) for item in tqdm.tqdm(ds.items)])

