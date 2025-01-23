import argparse
from einops.layers.torch import Rearrange
from glob import glob
import json
from libauc.losses.contrastive import GCLoss_v1
import math
import matplotlib.pyplot as plt
from mir_eval.key import weighted_score
from nnAudio.features.mel import MelSpectrogram
import numpy as np
import os
import pickle
import random
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, top_k_accuracy_score, r2_score
import torch
import torch.distributed.nn.functional as dist_fn
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
from vit_pytorch.simple_vit import Transformer


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter configuration')

    # Run parameters
    parser.add_argument('--run_name', type=str, help='Name of the run')
    parser.add_argument('--wandb_project', type=str, help='Weights and Biases project name')
    parser.add_argument('--wandb', action='store_true', help='Use Weights and Biases')

    # Dataset parameters
    parser.add_argument('--dataroot', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to use')
    parser.add_argument('--hop_samples', type=int, help='Number of samples to hop (overlapping window) for evaluation.')
    parser.add_argument('--sr', type=int, default=16000, help='Sampling rate of audio data')
    parser.add_argument('--filenames', type=str, help='JSON file containing a list of filenames (useful for very large datasets)')
    parser.add_argument('--unlabeled', action='store_true', help='Is the dataset unlabeled? (We assume a labeled dataset otherwise).')

    # Preprocessing parameters
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel bands')
    parser.add_argument('--log_mel', action='store_true', help='Take the log of the input spectrograms before passing them to the model')

    # DataLoader parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer to use (adam or sgd)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--lr_schedule', type=str, default='constant', choices=['constant', 'cosine'], help='Learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs if non-constant learning rate schedule is set')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--sogclr_tau', type=float, default=0.1, help='Temperature for SogCLR (contrastive)')
    parser.add_argument('--sogclr_gamma', type=float, default=0.9, help='Gamma for SogCLR (contrastive)')
    parser.add_argument('--sogclr_eps', type=float, default=1e-8, help='Epsilon value for SogCLR (contrastive)')
    parser.add_argument('--isogclr', action='store_true', help='Use iSogCLR for individualized temperatures')
    parser.add_argument('--gamma_schedule', type=str, default='constant', choices=['constant', 'cosine'], help='Gamma schedule for SogCLR. If cosine, decays from 1.0 to --sogclr_gamma.')
    parser.add_argument('--grad_clip', type=float, help='Gradient clipping (by norm)')

    # Training parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Checkpoint every x epochs')
    parser.add_argument('--resume', type=str, help='Resume training from the last checkpoint')
    parser.add_argument('--resume_optimizer', type=str, help='Load optimizer state from the last checkpoint')
    parser.add_argument('--resume_epochs', type=int, default=0, help='Epoch to resume training from (for checkpointing and learning rate scheduling). If provided, the script will run args.epochs - args.resume_epochs epochs of training.')
    parser.add_argument('--ignore_layers', type=str, nargs='*', default=[], help='List of layer names to ignore during loading from checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for deterministic output')
    parser.add_argument('--task_type', type=str, choices=['binary', 'multiclass', 'regression', 'contrastive', 'mae'], required=True, help='Task type for training (binary, multiclass, regression, contrastive, or mae [masked autoencoder])')
    parser.add_argument('--mask_ratio', type=float, default=None, help='Mask ratio for masked autoencoder task')
    parser.add_argument('--train_only_head_epochs', type=int, help='Number of epochs to train only the model head (freeze backbone for this many epochs)')
    parser.add_argument('--local-rank', type=int, help='Local rank (passed in by torchrun)')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Backend for distributed training (default: NCCL)')

    # Evaluation parameters 
    parser.add_argument('--key_detection', action='store_true', help='Additionally evaluate on key detection (must have num_outputs=24)')
    parser.add_argument('--fast_eval', action='store_true', help='Skip some parts of hyperparameter search for quicker evaluation (usually accurate, especially for MTAT, but 10x faster). Actual result is at least as good as the result of fast evaluation.')
    parser.add_argument('--eval_start_idx', type=int, default=0, help='Start evaluation script (grid search) from permutation index i (useful if a run fails)')
    parser.add_argument('--eval_save_embeds', type=str, help='Location to save model embeddings to (saved as a pickle file)')
    parser.add_argument('--eval_load_embeds', type=str, help='Location to load model embeddings to (expects a pickle file)')
    parser.add_argument('--eval_save_results', type=str, help='Location to save evaluation results to (JSON file)')
    parser.add_argument('--eval_patience', type=int, help='Patience for evaluation (default: none)')
    parser.add_argument('--use_other_patch_size', action='store_true', help='Use the additional patch size for evaluation')
    parser.add_argument('--eval_hybrid', action='store_true', help='If there is an additional patch size provided (see --additional_patch_size), this sets the model to compute both forward passes and concatenating them.')

    # Model parameters
    parser.add_argument('--patch_size', type=int, nargs='+', default=[16], help='Patch size for model input (can be a single integer or a list of two integers)')
    parser.add_argument('--additional_patch_size', type=int, nargs='*', help='Additioanl patch size for model input (can be omitted, a single integer, or a list of two integers)')
    parser.add_argument('--num_outputs', type=int, default=50, help='Number of model outputs')
    parser.add_argument('--dim', type=int, default=256, help='ViT model dimension')
    parser.add_argument('--depth', type=int, default=6, help='ViT number of layers')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Decoder depth for masked autoencoder task (default: 4)')
    parser.add_argument('--heads', type=int, default=16, help='ViT number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=1024, help='ViT MLP dimension')
    parser.add_argument('--dim_head', type=int, default=64, help='ViT attention head dimension')
    parser.add_argument('--proj_head', type=str, nargs='+', help='Projection head dimensions to add on to ViT model')
    parser.add_argument('--arch', type=str, help='ViT architecture', choices=['vit-s-32', 'vit-b-16', 'vit-l-16'])

    # Experimental parameters
    parser.add_argument('--mixup_alpha', type=float, help='Alpha parameter for MixUp augmentation')
    parser.add_argument('--mixup_beta', type=float, help='Beta parameter for MixUp augmentation')
    parser.add_argument('--max_frame_distance', type=int, help='Maximum frame distance from first view for frame selection of positives in contrastive learning (default: None (inf))')
    parser.add_argument('--add_layers', type=int, help='Number of transformer layers to add to the model (default: 0)')
    parser.add_argument('--unfreeze_last_n_layers', type=int, help='Unfreeze the last n transformer layers')


    args = parser.parse_args()

    # architecture
    if args.arch and args.arch.lower() == 'vit-s-32':
        # dim 384, depth 12, MLP 1536, 6 heads, 22M parameters
        args.dim = 384
        args.depth = 12
        args.mlp_dim = 1536
        args.heads = 6
    if args.arch and args.arch.lower() == 'vit-b-16':
        # dim 768, depth 12, MLP 3072, 12 heads, 87M parameters
        args.dim = 768
        args.depth = 12
        args.mlp_dim = 3072
        args.heads = 12
    if args.arch and args.arch.lower() == 'vit-l-16':
        # dim 1024, depth 24, MLP 4096, 16 heads, 303M parameters
        args.dim = 1024
        args.depth = 24
        args.mlp_dim = 4096
        args.heads = 16

    # if args.patch_size is a list of [single element], convert it to int
    # vit-pytorch only accepts tuples
    args.patch_size = tuple(args.patch_size) 
    if len(args.patch_size) == 1:
        args.patch_size = args.patch_size[0]

    if args.additional_patch_size:
        args.additional_patch_size = tuple(args.additional_patch_size) 
        if len(args.additional_patch_size) == 1:
            args.additional_patch_size = args.additional_patch_size[0]

    return args


class MelSpectrogramDataset(Dataset):
    ''' Dataset for pre-computed Mel Spectrograms '''
    def __init__(
        self, 
        dataroot: str, # path to dataset
        frame_size: int = None, # how many frames to return? None for entire spectrogram
        labeled: bool = True, # do pickle files contain (spec, label) pairs or just spec?
        pickle_extensions: list = ['.pkl'], # extensions to match when searching for files
        n_views: int = 1, # number of 'crops' of the spectrogram to return
        filenames = None, # path to JSON file containing filenames
        max_frame_distance: int = None # maximum number of frames far from the first starting point, sampled uniformly
    ):
        super().__init__()
        self.dataroot = dataroot
        self.frame_size = frame_size
        self.labeled = labeled
        self.n_views = n_views
        self.max_frame_distance = max_frame_distance
        
        # find files
        if filenames is None:
            self.filenames = []
            for ext in pickle_extensions:
                self.filenames.extend(glob(os.path.join(dataroot, '**/*' + ext), recursive=True))
        else:
            with open(filenames, 'r') as f:
                filenames = json.load(f)

            # filter filenames to this subset
            subset = os.path.basename(os.path.normpath(dataroot))
            self.filenames = [
                os.path.join(dataroot, *os.path.normpath(f).split(os.sep)[1:])
                for f in filenames 
                if os.path.normpath(f).startswith(subset)
            ]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, i: int):
        with open(self.filenames[i], 'rb') as f:
            data = pickle.load(f)
            
        if self.labeled:
            spec, label = data
        else:
            spec, label = data, 0 # no label
            
        if self.frame_size is not None:
            total_frames = spec.shape[-1]
            if total_frames < self.frame_size:
                raise ValueError(f'Spectrogram has fewer frames ({total_frames}) than the requested frame size ({self.frame_size}).')
            
            # Select the first frame uniformly
            start = random.randint(0, total_frames - self.frame_size)
            end = start + self.frame_size
            specs = [spec[..., start:end]]
            
            # Select subsequent frames based on the first one if n_views > 1
            for _ in range(1, self.n_views):
                if self.max_frame_distance is not None:
                    # Generate new start position uniformly within the range of max_frame_distance
                    new_start = start + random.randint(-self.max_frame_distance, self.max_frame_distance)
                    # Make sure the new start is within valid bounds
                    new_start = max(0, min(new_start, total_frames - self.frame_size))
                else:
                    # Uniform selection as fallback
                    new_start = random.randint(0, total_frames - self.frame_size)
                
                new_end = new_start + self.frame_size
                specs.append(spec[..., new_start:new_end])
            
            spec = specs if self.n_views > 1 else specs[0]
            
        return i, spec, label
    

@torch.no_grad()
def predict(model: nn.Module, spec: torch.Tensor, chunk_size: int):
    ''' Averages predictions over all chunks (of size chunk_size) '''

    # No batched inputs
    spec = spec.squeeze()
    assert spec.dim() == 2

    model.eval()
    n_mels, total_frames = spec.shape

    # Calculate the number of chunks we can extract
    n_chunks = total_frames // chunk_size

    # If there are no full chunks
    if n_chunks == 0:
        print("The input spectrogram is too small for the specified chunk size.")
        return None

    chunks = spec[:, :n_chunks * chunk_size]
    chunks = chunks.view(n_mels, n_chunks, chunk_size)
    chunks = chunks.permute(1, 0, 2).unsqueeze(1)

    outputs = model(chunks)
    return torch.mean(outputs, dim=0)


def compute_metrics(targets: np.ndarray, predictions: np.ndarray, args: argparse.Namespace):
    ''' Compute metrics for the given task type. Input shape: (batch_size, n_outputs) '''
    if args.task_type.lower() == 'binary':
        auroc = roc_auc_score(targets, predictions, average='macro')
        auprc = average_precision_score(targets, predictions, average='macro')

        return {
            'auroc': auroc,
            'auprc': auprc,
            'text': f'AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}'
        }

    elif args.task_type.lower() == 'multiclass':
        top_1_accuracy = accuracy_score(targets, np.argmax(predictions, axis=1))
        top_5_accuracy = top_k_accuracy_score(targets, predictions, k=5)

        metrics = {
            'top_1_accuracy': top_1_accuracy,
            'top_5_accuracy': top_5_accuracy,
            'text': f'Top-1 Accuracy: {top_1_accuracy:.4f}, Top-5 Accuracy: {top_5_accuracy:.4f}'
        }

        if args.key_detection:
            weighted_accuracy = key_detection_accuracy(targets, np.argmax(predictions, axis=1))
            metrics['weighted_accuracy'] = weighted_accuracy
            metrics['text'] += f', Weighted: {weighted_accuracy:.4f}'

        return metrics

    elif args.task_type.lower() == 'regression':
        r2 = r2_score(targets, predictions)

        metrics = {
            'r2': r2,
            'text': f'R^2: {r2:.4f}'
        }

        dims = targets.shape[1]
        if dims > 1:
            for i in range(targets.shape[1]):
                r2_dim = r2_score(targets[:, i], predictions[:, i])
                metrics[f'r2_dim_{i}'] = r2_dim
                metrics['text'] += f', R^2_dim_{i}: {r2_dim:.4f}'
        
        return metrics
    
    elif args.task_type.lower() in ['contrastive', 'mae']:
        return {'text': ''} # no metrics

    else:
        raise Exception(f'Task type {args.task_type} not supported')


def compute_loss(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, criterion: nn.Module, args: argparse.Namespace, return_mae_outputs: bool = False, recurse: bool = True):
    # evaluate for both patch sizes if there are two patch sizes
    if args.additional_patch_size is not None and recurse:
        outputs, loss_a = compute_loss(model, inputs, labels, indices, criterion, args, recurse = False)
        model.toggle_embeddings()
        _, loss_b = compute_loss(model, inputs, labels, indices, criterion, args, recurse = False)
        model.toggle_embeddings()
        loss = (loss_a + loss_b) / 2

    elif args.task_type.lower() in ['binary', 'multiclass', 'regression']:
        if args.mixup_alpha is not None:
            inputs, y_a, y_b, lam = mixup_data(inputs, labels, args.mixup_alpha, args.mixup_beta)

        if args.mask_ratio:
            outputs = masked_model_output(inputs, model, args)
        else:
            outputs = model(inputs)

        if args.mixup_alpha is not None:
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        loss = criterion(outputs, labels)

    elif args.task_type.lower() in ['contrastive']:
        assert args.mask_ratio is not None, 'You must provide --mask_ratio for contrastive learning.'
        a, b = inputs

        if args.mixup_alpha is not None:
            a = mixup_data(a, None, args.mixup_alpha, args.mixup_beta)[0]
            b = mixup_data(b, None, args.mixup_alpha, args.mixup_beta)[0]

        za = masked_model_output(a, model, args)
        zb = masked_model_output(b, model, args)

        # distributed training
        if args.world_size > 1:
            # all gather using dist.nn.functional which allows autograd tracking
            gathered_za = dist_fn.all_gather(za)
            gathered_zb = dist_fn.all_gather(zb)
            gathered_indices = dist_fn.all_gather(indices)

            # concatenate the gathered outputs
            za = torch.cat(gathered_za, dim=0)
            zb = torch.cat(gathered_zb, dim=0)
            indices = torch.cat(gathered_indices, dim=0)

        outputs = torch.zeros(1) # no output
        loss = criterion(za, zb, indices.cpu())

    elif args.task_type.lower() in ['mae']:
        assert args.mask_ratio is not None, 'You must provide --mask_ratio for masked autoencoding.'

        x = model.to_patch_embedding(inputs)
        x += model.pos_embedding.to(inputs.device, dtype=inputs.dtype)

        B, N, _ = x.shape
        n_masked = int(args.mask_ratio * N)
        indices = torch.stack([torch.randperm(N) for _ in range(B)])
        mask_indices = indices[:, :n_masked].to(args.device)
        unmask_indices = indices[:, n_masked:].to(args.device)

        unmasked = x.gather(1, unmask_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        encoded = model.transformer(unmasked)

        mask_tokens = model.mask_token.repeat(B, n_masked, 1)
        mask_tokens += model.pos_embedding[mask_indices.cpu()].to(args.device)
        decoder_input = torch.cat([encoded, mask_tokens], dim=1)

        combined_indices = torch.cat([unmask_indices, mask_indices], dim=1)
        sorted_indices = torch.argsort(combined_indices, dim=1).to(args.device)
        decoder_input = decoder_input.gather(1, sorted_indices.unsqueeze(-1).expand(-1, -1, decoder_input.size(-1)))

        decoded = model.decoder(decoder_input)
        decoded = model.decoder_norm(decoded)
        outputs = model.decoder_head(decoded)

        outputs_masked = outputs.gather(1, mask_indices.unsqueeze(-1).expand(-1, -1, outputs.size(-1)))

        actual_patched = model.patchify(inputs)
        actual = actual_patched.gather(1, mask_indices.unsqueeze(-1).expand(-1, -1, actual_patched.size(-1)))

        outputs = outputs.scatter(1, unmask_indices.unsqueeze(-1).expand(-1, -1, outputs.size(-1)), torch.log1p(actual_patched).gather(1, unmask_indices.unsqueeze(-1).expand(-1, -1, actual_patched.size(-1))))
        outputs = model.unpatchify(outputs) if return_mae_outputs else torch.zeros(1)

        loss = criterion(outputs_masked, torch.log1p(actual))

    else:
        raise Exception(f'Task type {args.task_type} not supported.')
    
    return outputs, loss


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epoch: int, args: argparse.Namespace):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    if args.world_size > 1:
        train_loader.sampler.set_epoch(epoch)

    set_schedulers(epoch, criterion, optimizer, args)

    progress_bar = tqdm(train_loader) if args.rank == 0 else train_loader
    for indices, inputs, labels in progress_bar:
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
        if args.log_mel: 
            inputs = torch.log1p(inputs)
        indices = indices.to(args.device)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        outputs, loss = compute_loss(model, inputs, labels, indices, criterion, args)
        running_loss += loss.item()

        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        all_targets.append(labels.cpu().numpy())
        all_predictions.append(outputs.detach().cpu().numpy())
        
        if args.rank == 0:
            progress_bar.set_description(f'Loss: {loss.item():.4f}')

    # Concatenate all stored targets and predictions
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Compute metrics
    train_metrics = compute_metrics(all_targets, all_predictions, args)

    avg_loss = running_loss / len(train_loader)

    if args.rank == 0:
        print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train {train_metrics["text"]}')

    return avg_loss, train_metrics


def test(model: nn.Module, test_dataset: Dataset, criterion: nn.Module, epoch: int, args: argparse.Namespace):
    # don't test on contrastive pretraining
    if args.task_type == 'contrastive':
        return 0, {'text': ''}
    
    # just output examples for MAE
    if args.task_type == 'mae':
        output_mae_examples(model, test_dataset, epoch, args)
        return 0, {'text': ''}
    
    model.eval()
    all_targets = []
    all_predictions = []
    running_loss = 0.0

    progress_bar = tqdm(test_dataset)
    with torch.no_grad():
        for _, inputs, labels in progress_bar:
            if isinstance(inputs, list):
                inputs = torch.stack(inputs, dim=0)
            if args.log_mel:
                inputs = torch.log1p(inputs)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = predict(model, inputs, chunk_size=args.mel_frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_targets.append(labels.unsqueeze(0).cpu().numpy())
            all_predictions.append(outputs.unsqueeze(0).cpu().numpy())
            
            progress_bar.set_description(f'Loss: {loss.item():.4f}')

    # Concatenate all stored targets and predictions
    if labels.numel() == 1:
        all_targets = np.array(all_targets).squeeze(1)
        all_predictions = np.array(all_predictions).squeeze(1)
    else:
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

    # Compute metrics
    test_metrics = compute_metrics(all_targets, all_predictions, args)
    avg_loss = running_loss / len(test_dataset)
    test_metrics_str = f', Test {test_metrics["text"]}' if len(test_metrics["text"]) > 0 else ''

    if args.rank == 0:
        print(f'Test Loss: {avg_loss:.4f}{test_metrics_str}')

    return avg_loss, test_metrics


def get_dataset(dataroot: str, args: argparse.Namespace, distributed=False, rank=0, world_size=1, drop_last=True):
    dataset = MelSpectrogramDataset(
        dataroot=dataroot,
        frame_size=args.mel_frames,
        labeled=not args.unlabeled,
        n_views=1 if args.task_type != 'contrastive' else 2,
        filenames=args.filenames,
        max_frame_distance=args.max_frame_distance
    )

    # Use DistributedSampler if training in distributed mode
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last
    )

    return dataset, dataloader


def get_n_frames(n_samples: int, args: argparse.Namespace):
    ''' How many frames is n_samples samples? '''
    mel_spectrogram = MelSpectrogram(sr=args.sr, n_mels=128, verbose=False)

    # patch size along the time dimension
    patch_size_time = args.patch_size if isinstance(args.patch_size, int) else args.patch_size[1]

    mel_frames = mel_spectrogram(torch.randn(1, 1, n_samples)).shape[-1]
    mel_frames = math.floor(mel_frames / patch_size_time) * patch_size_time
    return mel_frames


def get_optimizer(model: nn.Module, args: argparse.Namespace):
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=args.sgd_momentum, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer. Please use 'sgd' or 'adam'.")
    
    return optimizer


def get_criterion(args: argparse.Namespace, N: int):
    if args.task_type.lower() == 'binary':
        return nn.BCEWithLogitsLoss()

    elif args.task_type.lower() == 'multiclass':
        return nn.CrossEntropyLoss()

    elif args.task_type.lower() in ['regression', 'mae']:
        return nn.MSELoss()
    
    elif args.task_type.lower() == 'contrastive':
        return GCLoss_v1(
            N=N, 
            tau=args.sogclr_tau, 
            gamma=args.sogclr_gamma, 
            gamma_schedule=args.gamma_schedule,
            device=args.device,
            distributed=False, # args.world_size > 1,
            gamma_decay_epochs=args.epochs,
            eps=args.sogclr_eps,
            enable_isogclr=args.isogclr
        )

    else:
        raise Exception(f'Task type {args.task_type} not supported')


def seed_everything(seed: int):
    # Seed the built-in random module
    random.seed(seed)
    
    # Seed numpy
    np.random.seed(seed)
    
    # Seed torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    
    # Ensure deterministic behavior when using torch.backends.cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: nn.Module, checkpoint_dir: str, filename: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    model = model.module if isinstance(model, DDP) else model
    torch.save(model.state_dict(), checkpoint_path)


def load_model(model: nn.Module, checkpoint_path: str, device: str, ignore_layers: list, verbose: bool):
    '''
    Load model from checkpoint. Ignores (does not load) weights for 
    layers whose names start with any string in ignore_layers.
    '''
    checkpoint = torch.load(checkpoint_path, map_location=device)

    filtered_state_dict = {
        k: v for k, v in checkpoint.items() 
        if not any(k.startswith(layer) for layer in ignore_layers)
    }

    model.load_state_dict(filtered_state_dict, strict=False)

    if ignore_layers and verbose:
        print(f'==> Loaded model from {checkpoint_path}, ignoring layers: {", ".join(ignore_layers)}')


def save_optimizer(optimizer: optim.Optimizer, criterion: nn.Module, checkpoint_dir: str, filename: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    data = {
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if isinstance(criterion, GCLoss_v1):
        data['sogclr_u'] = criterion.u

    torch.save(data, checkpoint_path)


def load_optimizer(optimizer: optim.Optimizer, criterion: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'==> Optimizer state loaded from {checkpoint_path}')

    if isinstance(criterion, GCLoss_v1) and 'sogclr_u' in checkpoint:
        criterion.u = checkpoint['sogclr_u']
        print(f'==> SogCLR parameter loaded from {checkpoint_path}')


def add_proj_head(model: nn.Module, proj_head: list):
    ''' 
    Adds MLP projection head on top of the model, where proj_head are the layer output dimensions and dropout layers are specified in the format "d<probability>" (e.g., "d0.5"). 
    '''
    
    mlp_dimensions = [model.linear_head.in_features] + proj_head
    layers = []
    current_dim = model.linear_head.in_features
    
    for output_dim in mlp_dimensions[1:]:
        # handle dropout layers
        if output_dim.lower().startswith('d'):
            try:
                dropout_prob = float(output_dim[1:])
                layers.append(nn.Dropout(dropout_prob))
            except ValueError:
                raise ValueError(f'Invalid dropout probability: {output_dim}')
            
        # handle linear layers
        else:
            input_dim = current_dim
            output_dim = int(output_dim)
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            current_dim = output_dim
    
    # remove the last relu, if added
    if layers and isinstance(layers[-1], nn.ReLU):
        layers.pop()
    
    model.linear_head = nn.Sequential(*layers)


def setup_for_mae(model: nn.Module, args: argparse.Namespace):
    patch_height, patch_width = args.patch_size if isinstance(args.patch_size, tuple) else (args.patch_size, args.patch_size)
    h = args.n_mels // patch_height
    w = args.mel_frames // patch_width

    model.mask_token = nn.Parameter(torch.randn(1, 1, args.dim))
    model.decoder = Transformer(args.dim, args.decoder_depth, args.heads, 64, args.mlp_dim)
    model.decoder_norm = nn.LayerNorm(args.dim)

    model.patchify = Rearrange(
        'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
        p1 = patch_height, p2 = patch_width
    )
    model.unpatchify = Rearrange(
        'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
        h = h, w = w, p1 = patch_height, p2 = patch_width
    )
    model.decoder_head = nn.Linear(args.dim, patch_height * patch_width)


def freeze_unfreeze_backbone(model: nn.Module, freeze: bool):
    ''' Freeze or unfreeze all model parameters except mlp head '''
    for name, param in model.named_parameters():
        if name.startswith('linear_head'):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze


def key_detection_accuracy(targets: list, predictions: list):
    keys = [
        'c major', 'c minor', 'db major', 'db minor', 'd major', 
        'd minor', 'eb major', 'eb minor', 'e major', 'e minor', 
        'f major', 'f minor', 'gb major', 'gb minor', 'g major', 
        'g minor', 'ab major', 'ab minor', 'a major', 'a minor', 
        'bb major', 'bb minor', 'b major', 'b minor'
    ]

    targets = [keys[i] for i in targets]
    predictions = [keys[i] for i in predictions]

    score = np.mean([weighted_score(target, pred) for target, pred in zip(targets, predictions)])
    return score


def output_mae_examples(model: nn.Module, test_dataset: Dataset, epoch: int, args: argparse.Namespace):
    if not args.checkpoint_dir or ((epoch+1) % args.checkpoint_epochs != 0):
        return
    
    n_imgs = 10

    indices = torch.randint(0, len(test_dataset), (n_imgs,))
    inputs = torch.stack([test_dataset[i][1] for i in indices]).to(args.device)
    outputs, _ = compute_loss(model, inputs, None, None, nn.MSELoss(), args, return_mae_outputs=True)

    img_dir = os.path.join(args.checkpoint_dir, 'outputs', f'epoch_{epoch:03}')
    os.makedirs(img_dir, exist_ok=True)

    for i in range(n_imgs):
        input_spectrogram_np = torch.log1p(inputs[i]).squeeze().cpu().detach().numpy()
        output_spectrogram_np = outputs[i].squeeze().cpu().detach().numpy()

        _, axes = plt.subplots(1, 2, figsize=(15, 5))

        im1 = axes[0].imshow(input_spectrogram_np, aspect='auto', origin='lower')
        axes[0].set_title(f'Input Spectrogram {i}')
        axes[0].set_ylabel('Frequency bins')
        axes[0].set_xlabel('Time frames')
        plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')

        im2 = axes[1].imshow(output_spectrogram_np, aspect='auto', origin='lower')
        axes[1].set_title(f'Output Spectrogram {i}')
        axes[1].set_ylabel('Frequency bins')
        axes[1].set_xlabel('Time frames')
        plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')

        file_path = os.path.join(img_dir, f'input_output_spectrogram_{i}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

    if args.rank == 0:
        print(f'==> Saved {n_imgs} input-output spectrogram images in {img_dir}')


def mask_inputs(x: torch.Tensor, mask_ratio: float, device: str):
    ''' Input masking for contrastive learning '''
    # input B, N, D --> output B, N * (1 - mask_ratio), D
    B, N, _ = x.shape
    n_masked = int(mask_ratio * N)
    indices = torch.stack([torch.randperm(N) for _ in range(B)])
    unmask_indices = indices[:, n_masked:].to(device)
    unmasked = x.gather(1, unmask_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    return unmasked


def masked_model_output(x: torch.Tensor, model: nn.Module, args: argparse.Namespace):
    ''' ViT forward pass with masking. '''
    model = model.module if isinstance(model, DDP) else model

    x = model.to_patch_embedding(x)
    x += model.pos_embedding.to(x.device, dtype=x.dtype)

    x = mask_inputs(x, args.mask_ratio, args.device)

    z = model.transformer(x)
    z = z.mean(dim=1)
    z = model.to_latent(z)
    z = model.linear_head(z)

    return z


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha=5.0, beta=2.0):
    ''' Compute the mixup data. Return mixed inputs, pairs of targets, and lambda '''
    lam = np.random.beta(alpha, beta) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = (y, y[index]) if y is not None else (None, None)
    return mixed_x, y_a, y_b, lam


def set_schedulers(epoch: int, criterion: nn.Module, optimizer: optim.Optimizer, args: argparse.Namespace):
    # gamma schedule
    if isinstance(criterion, GCLoss_v1):
        criterion.adjust_gamma(epoch)

        if args.rank == 0:
            print(f'Adjusted gamma according to schedule: {criterion.gamma:.5f}')

    # learning rate schedule
    if args.lr_schedule.lower() == 'cosine':
        # warmup
        if epoch < args.warmup_epochs:
            lr = args.learning_rate * float(epoch + 1) / args.warmup_epochs
        
        # cosine decay
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr