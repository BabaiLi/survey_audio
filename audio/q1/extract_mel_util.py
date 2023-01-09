import math
import argparse
from pathlib import Path, PosixPath
from functools import partial

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from torch_stft import STFT
from joblib import Parallel, delayed
from librosa.filters import mel as librosa_mel_fn

def extract_mel(
    inputs: PosixPath, 
    output_dir: PosixPath, 
    stft: torch.nn.Module, 
    mel_basis: torch.Tensor, 
    device: str, 
    save_to_np: bool, 
    mel: int, 
    mfcc: int,
    ):
    
    audio, sr = sf.read(inputs)
    audio     = torch.FloatTensor(audio).to(device).unsqueeze(0)

    magnitude, phase = stft.transform(audio)
    mel_output       = torch.matmul(mel_basis, magnitude)
    melspec          = mel_output.clamp(min=1e-5).log().squeeze().cpu()
    
    if mfcc:
        n = torch.arange(float(n_mel_channels))
        k = torch.arange(float(mfcc)).unsqueeze(1)
        dct = torch.cos(math.pi / n_mel_channels * (n + 0.5) * k)
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mel_channels))
        melspec = torch.matmul(melspec.unsqueeze(0).transpose(-1, -2), dct.t()).transpose(-1, -2)
    
    output = Path(output_dir, inputs.parent.name, inputs.stem)
    if not output.parent.exists():
        Path.mkdir(output.parent, exist_ok=True)
    print(f"Save to {output}")
    
    if save_to_np:
        np.save(output, melspec.numpy())
    else:
        output = Path(f"{output}.pt")
        torch.save(melspec, output)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Declare I/O Path
    inputs_path = Path(args.inputs)
    if inputs_path.is_dir():
        inputs_list = list(inputs_path.rglob('*.wav'))
    else:
        inputs_list = [inputs_path]
        
    output_dir  = Path(args.output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)
    
    # Define Mel bands.
    mel_basis = librosa_mel_fn(
        sr     = 22050, 
        n_fft  = args.filter_length, 
        n_mels = args.n_mel_channels, 
        fmin   = args.f_min, 
        fmax   = args.f_max
    )
    mel_basis = torch.from_numpy(mel_basis).float().to(device)
    
    # Define STFT parameters.
    stft = STFT(
        filter_length = args.filter_length, 
        hop_length    = args.hop_length, 
        win_length    = args.win_length,
        window        = args.window
    ).to(device)
    
    # Fixed the partial parameters of Mel Function.
    mel_fn = partial(
        extract_mel,
        output_dir = output_dir,
        stft       = stft, 
        mel_basis  = mel_basis,
        device     = device,
        save_to_np = args.save_to_np,
        mel        = args.n_mel_channels,
        mfcc       = args.n_mfcc_channels,
    )
        
    # Start
    Parallel(n_jobs=8)(delayed(mel_fn)(i) for i in tqdm(inputs_list, desc="Preprocessing", total=len(inputs_list)))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str)
    parser.add_argument("--output_dir",      type=str, default="extract_mel")
    parser.add_argument("--filter_length",   type=int, default=1024)
    parser.add_argument("--win_length",      type=int, default=1024)
    parser.add_argument("--hop_length",      type=int, default=256)
    parser.add_argument("--n_mel_channels",  type=int, default=80)
    parser.add_argument("--n_mfcc_channels", type=int, default=0)
    parser.add_argument("--f_min",           type=float, default=20.0)
    parser.add_argument("--f_max",           type=float, default=8000.0)
    parser.add_argument("--window",          type=str, default='hann')
    parser.add_argument("--save_to_np",      action='store_true')
    args = parser.parse_args()
    
    main(args)
