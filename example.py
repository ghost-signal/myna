'''
Minimal script example for model inference
'''

from argparse import Namespace
from nnAudio.features.mel import MelSpectrogram
import torch
import torchaudio
import torchaudio.transforms as T

from utils import get_n_frames, load_model
from vit import SimpleViT

FILENAME = 'your_file_here.wav' # file to get embeddings from
MODEL_PATH = 'myna-hybrid.pth' # path to model checkpoint
MODEL_TYPE = 'hybrid' # 'square', 'vertical', or 'hybrid'
HYBRID_MODE = True # concatenate embeddings for hybrid models; disable to only use square patches
N_SAMPLES = 50000 # number of samples per embedding
MYNA_SR = 16000 # myna constant


def load_and_preprocess_audio(filename: str):
    # load audio
    signal, sr = torchaudio.load(filename)
    
    # make mono if necessary
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    
    # resample to target sample rate
    if sr != MYNA_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=MYNA_SR)
        signal = resampler(signal)

    # sanity check
    assert signal.dim() == 2

    # compute spectrogram
    mel_spec = MelSpectrogram(sr=16000, n_mels=128, verbose=False)
    ms = mel_spec(signal)
    
    return ms

def batch_spectrogram(ms: torch.Tensor, n_frames: int):
    # sanity check
    assert ms.dim() == 3 and ms.shape[0] == 1

    # discard excess frames
    num_chunks = ms.shape[-1] // n_frames
    ms = ms[:, :, :num_chunks * n_frames]

    # split the tensor into chunks and stack them
    chunks = torch.chunk(ms, num_chunks, dim=2)
    batch = torch.stack(chunks)

    return batch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patch_size = (128, 2) if MODEL_TYPE == 'vertical' else 16

# sanity check
if HYBRID_MODE:
    assert MODEL_TYPE == 'hybrid', 'hybrid mode can only be enabled for hybrid model types'

# number of spectrogram frames to feed into the model
n_frames = get_n_frames(
    n_samples=N_SAMPLES, 
    args=Namespace(
        sr=16000, 
        patch_size=patch_size
    )
)

# initialize model first
model = SimpleViT(
    image_size=(128, n_frames),
    channels=1,
    patch_size=patch_size,
    num_classes=50, # doesn't matter
    dim=384,
    depth=12,
    heads=6,
    mlp_dim=1536,
    additional_patch_size=(128, 2) if MODEL_TYPE == 'hybrid' else None
)

# now load weights
load_model(model, MODEL_PATH, device, ignore_layers=['linear_head'], verbose=True)
model.linear_head = torch.nn.Identity()
model.hybrid_mode = HYBRID_MODE

# load and preprocess audio
ms = load_and_preprocess_audio(FILENAME)
ms = batch_spectrogram(ms, n_frames)

# forward pass
model.eval()
with torch.no_grad():
    embeds = model(ms)

print(f'Successfully computed embeddings of shape: {embeds.shape}')