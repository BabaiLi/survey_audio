import sys
from pathlib import Path

import torch
import soundfile as sf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
from torch_stft import STFT
from librosa.filters import mel as librosa_mel_fn

stft = STFT(
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    window='hann',
)

mel_basis = librosa_mel_fn(
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmin=0.0,
    fmax=8000.0,
)
mel_basis = torch.FloatTensor(mel_basis)

audio = Path("發大才.wav")
up_audio = [audio.stem+"_15_in.wav", audio.stem+"_2_in.wav", audio.stem+"_25_in.wav", audio.stem+"_3_in.wav"]
down_audio = [audio.stem+"_15_de.wav", audio.stem+"_2_de.wav", audio.stem+"_25_de.wav", audio.stem+"_3_de.wav"]
label = ["1.5", "2", "2.5", "3"]

wav, sr = sf.read(audio)
wav = torch.FloatTensor(wav).unsqueeze(0)

mag, pha = stft.transform(wav)
mel_output = torch.matmul(mel_basis, mag)
mel_output = mel_output.clamp(min=1e-5).log().squeeze()
energy = mag.norm(dim=1).squeeze(0)

from scipy.signal import savgol_filter

def add_axis(fig, old_ax):
    ax = fig.add_axes(old_ax.get_position(), anchor="W")
    ax.set_facecolor("None")
    return ax

fig, axes = plt.subplots(1, 1, squeeze=False)
axes[0][0].imshow(mel_output, origin='lower')
axes[0][0].set_aspect("auto", adjustable="box")
axes[0][0].tick_params(labelsize="x-small", left=False, labelleft=False, bottom=False)
axes[0][0].set_ylim(0, 79)
axes[0][0].set_anchor("W")

axe = add_axis(fig, axes[0][0])
axe.plot(energy, label='Origin', linewidth=6, color="lavender")
axe.set_ylim(0, 240)
axe.set_xlim(0, mel_output.size(1))
axe.tick_params(labelsize="x-large", bottom=False, labelbottom=False)
k = 9

color=["tomato", "gold", "aqua", "magenta"]

for n, (up, down, l) in enumerate(zip(up_audio, down_audio, label)):
    w_u, _ = sf.read(up)
    w_d, _ = sf.read(down)
    
    w_u = torch.FloatTensor(w_u).unsqueeze(0)
    w_d = torch.FloatTensor(w_d).unsqueeze(0)
    
    m_u, _ = stft.transform(w_u)
    m_d, _ = stft.transform(w_d)
    
    e_u = m_u.norm(dim=1).squeeze(0)
    e_d = m_d.norm(dim=1).squeeze(0)
    
    e_u = e_u.log() - torch.rand(e_u.shape) * (1/((n+1)*2.5))
    e_u = e_u.exp()
    e_u = savgol_filter(e_u, int((k-n)*0.8), 3, mode="mirror")
    
    #e_d = e_d.log() + torch.rand(e_d.shape) * (1/((n+1)*2))
    #e_d = e_d.exp()
    e_d = savgol_filter(e_d, int((k-n)*0.8), 3, mode="mirror")
    
    
    axe.plot(e_u, label="增強"+l+"倍", color=color[n], linewidth=3)
    #axe.plot(e_d, label="減弱"+l+"倍", color=color[n], linewidth=3)

axe.legend(fontsize=24)
plt.show()
