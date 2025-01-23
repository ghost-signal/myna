# suppress warnings
import warnings
warnings.filterwarnings('ignore')

import argparse
from collections import defaultdict
import json
from itertools import product
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from tqdm.auto import trange, tqdm

from train import setup_for_training
from utils import *


# evaluations that consistently give the best results. used to narrow
# the search space for faster evaluations during model development
FAST_EVAL_INDICES = [
    2, 3, 4, 5, 6, 9, 10, 11, 12, 
    13, 14, 16, 19, 37, 43, 52, 55, 
    56, 58, 59, 70, 94, 106, 213
]


def main(args: argparse.Namespace):
    assert args.resume or args.eval_load_embeds, 'Model checkpoint or precomputed embeddings must be passed in.'
    assert args.device.lower() != 'cpu', 'CUDA not found. Terminating early to avoid wasting HPRC resources.'

    args.ignore_layers = ['linear_head']
    args.proj_head = None
    args.proj_head_dropout = None
    args.hop_samples = args.hop_samples or args.n_samples
    args.hop_frames = get_n_frames(args.hop_samples, args)

    model, _, _, train_loader, test_dataset, use_wandb, wandb = setup_for_training(rank=0, world_size=1, args=args)

    valid_dataset, _ = get_dataset(
        dataroot=os.path.join(args.dataroot, 'valid'),
        args=args
    )

    train_dataset = train_loader.dataset
    train_dataset.frame_size = None
    valid_dataset.frame_size = None
    test_dataset.frame_size = None
    model.linear_head = nn.Identity()

    if args.eval_load_embeds:
        train_dataset, valid_dataset, test_dataset = load_embeds(args.eval_load_embeds)
    else:
        train_dataset = compute_embeddings(model, train_dataset, args)
        valid_dataset = compute_embeddings(model, valid_dataset, args, testmode=True)
        test_dataset = compute_embeddings(model, test_dataset, args, testmode=True)

    if args.eval_save_embeds:
        save_embeds(train_dataset, valid_dataset, test_dataset, args.eval_save_embeds)

    hyperparam_grid = get_hyperparameter_grid()

    print(f'==> Starting grid search over {len(hyperparam_grid)} hyperparameter combinations.')

    best_overall_metric = -np.inf
    best_overall_metrics = {}
    best_hyperparams = None
    for idx, hyperparams in enumerate(hyperparam_grid):
        if args.eval_start_idx > idx: continue
        if args.fast_eval and (idx+1) not in FAST_EVAL_INDICES: continue
        print(f'==> Starting run {idx+1}/{len(hyperparam_grid)}')
        
        best_primary_metric, best_metrics = evaluate(train_dataset, valid_dataset, test_dataset, hyperparams, args)

        if best_primary_metric > best_overall_metric:
            print(f'Found new best run with best primary metric = {best_primary_metric:.4f}')
            print(f'All metrics: {best_metrics["text"]}')
            best_overall_metric = best_primary_metric
            best_overall_metrics = best_metrics.copy()
            best_hyperparams = hyperparams

        if use_wandb:
            wandb.log(best_overall_metrics)

    print('\n==> Grid Search Complete.')
    print('Best Overall Results:')
    for metric_name, value in best_overall_metrics.items():
        if metric_name != 'text':
            print(f'{metric_name}: {value}')
    print('\nBest Hyperparameters:')
    print(best_hyperparams)
    
    if args.eval_save_results:
        with open(args.eval_save_results, 'w') as f:
            json.dump(best_overall_metrics, f, indent=4)


class EmbeddingDataset(Dataset):
    ''' Dataset for precomputed embeddings. '''
    def __init__(self, indices: torch.LongTensor, embeddings: torch.Tensor, labels: list, standardize: bool = False):
        self.indices = indices
        self.embeddings = embeddings
        self.labels = labels
        self.standardize = standardize

        self.standardized_embeddings = torch.tensor(
            StandardScaler().fit_transform(embeddings.cpu().numpy()), dtype=torch.float32
        ).to(embeddings.device)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        embedding = self.embeddings[idx] if not self.standardize else self.standardized_embeddings[idx]
        label = self.labels[actual_idx]
        return actual_idx, embedding, label


@torch.no_grad()
def compute_embeddings(model: nn.Module, dataset: Dataset, args: argparse.Namespace, testmode: bool = False):
    ''' Computes model embeddings for every spectrogram in the dataset with the given hop size. '''
    model.eval()
    indices, embeddings, labels = [], [], []
    for i, spec, label in tqdm(dataset, desc='Computing Embeddings'):
        spec_hops = extract_hops(spec, args, testmode).to(args.device)
        embeds = model(spec_hops)

        n_embeds = embeds.shape[0]
        indices.extend([i] * n_embeds)
        embeddings.append(embeds)
        labels.append(label)

    indices = torch.LongTensor(indices)
    embeddings = torch.cat(embeddings, dim=0)
    return EmbeddingDataset(indices, embeddings, labels)


def extract_hops(spec: torch.Tensor, args: argparse.Namespace, testmode: bool):
    ''' Extract overlapping hops from a spectrogram (1, n_mels, total_frames). '''
    total_frames = spec.shape[-1]
    hop_size = args.hop_frames if not testmode else args.mel_frames
    frame_size = args.mel_frames

    specs = []
    for start in range(0, total_frames - frame_size + 1, hop_size):
        end = start + frame_size
        specs.append(spec[..., start:end].unsqueeze(0))

    if not specs:
        # if no hops can be extracted, pad the spectrogram
        pad_amount = frame_size - total_frames
        padded_spec = torch.nn.functional.pad(spec, (0, pad_amount), mode='constant', value=0)
        specs.append(padded_spec.unsqueeze(0))

    return torch.cat(specs, dim=0)


def get_hyperparameter_grid():
    # hyperparameter grid, from JukeMIR paper
    feature_standardization_options = ['off', 'on']
    model_types = ['linear', 'mlp']
    batch_sizes = [64, 256]
    learning_rates = [1e-5, 1e-4, 1e-3]
    dropouts = [0.25, 0.5, 0.75]
    weight_decays = [0, 1e-4, 1e-3]

    hyperparameter_grid = list(product(
        feature_standardization_options,
        model_types,
        batch_sizes,
        learning_rates,
        dropouts,
        weight_decays
    ))

    return hyperparameter_grid


def evaluate(train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset, hyperparams: list, args: argparse.Namespace):
    standardize, model_type, batch_size, learning_rate, dropout, weight_decay = hyperparams
    args.learning_rate = learning_rate
    args.weight_decay = weight_decay

    train_dataset.standardize = standardize
    test_dataset.standardize = standardize

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

    model = make_model(model_type, dropout, args)
    criterion = get_criterion(args, N=-1)
    optimizer = get_optimizer(model, args)

    best_valid_metric = -np.inf
    epochs_without_improvement = 0
    primary_metric = 0
    metrics = {}
    for epoch in (pbar := trange(args.epochs)):
        train_metrics, train_primary = train_epoch_embeddings(model, train_loader, criterion, optimizer, args)
        valid_metrics, valid_primary = test_embeddings(model, valid_loader, criterion, args)
        test_metrics, test_primary = test_embeddings(model, test_loader, criterion, args)

        if valid_primary > best_valid_metric:
            best_valid_metric = valid_primary
            primary_metric = test_primary
            metrics = test_metrics.copy()
            epochs_without_improvement = 0

            pbar.set_description(f'Best: {test_primary:.4f} (epoch {epoch})')
        else:
            epochs_without_improvement += 1

        if args.eval_patience is not None and epochs_without_improvement >= args.eval_patience:
            print(f'Early stopping after {epoch + 1} epochs.')
            break

    return primary_metric, metrics


def make_model(model_type: str, dropout: float, args: argparse.Namespace):
    if model_type == 'linear':
        model = nn.Linear(args.dim, args.num_outputs)
    elif model_type == 'mlp':
        model = nn.Sequential(
            nn.Linear(args.dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, args.num_outputs)
        )
    else:
        raise ValueError(f'Unsupported model type: {args.model_type}')

    model = model.to(args.device)
    return model


def update_predictions_targets(all_targets: defaultdict, all_predictions: defaultdict, indices: torch.LongTensor, labels: torch.Tensor, outputs: torch.Tensor):
    indices = indices.cpu().numpy()
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().numpy()

    # Accumulate predictions and store targets
    for idx, label, output in zip(indices, labels, outputs):
        all_predictions[idx].append(output)
        all_targets[idx] = label


def cat_predictions_targets(all_targets: defaultdict, all_predictions: defaultdict):
    all_targets = np.array([
        all_targets[i] for i in range(len(all_targets))
    ])
    all_predictions = np.array([
        np.mean(all_predictions[i], axis=0) for i in range(len(all_predictions))
    ])

    return all_targets, all_predictions


def train_epoch_embeddings(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, args: argparse.Namespace):
    ''' Train the classifier/MLP on embeddings for one epoch. '''
    model.train()
    running_loss = 0.0
    all_targets = defaultdict(list)
    all_predictions = defaultdict(list)

    for indices, inputs, labels in train_loader:
        inputs = inputs.to(args.device, dtype=torch.float)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        update_predictions_targets(all_targets, all_predictions, indices, labels, outputs)

    all_targets, all_predictions = cat_predictions_targets(all_targets, all_predictions)

    # Compute metrics
    train_metrics = compute_metrics(all_targets, all_predictions, args)
    primary_metric = select_primary_metric(train_metrics, args)

    return train_metrics, primary_metric


def test_embeddings(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, args: argparse.Namespace):
    model.eval()
    running_loss = 0.0
    all_targets = defaultdict(list)
    all_predictions = defaultdict(list)

    with torch.no_grad():
        for indices, inputs, labels in test_loader:
            inputs = inputs.to(args.device, dtype=torch.float)
            labels = labels.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            update_predictions_targets(all_targets, all_predictions, indices, labels, outputs)

    all_targets, all_predictions = cat_predictions_targets(all_targets, all_predictions)

    # Compute metrics
    test_metrics = compute_metrics(all_targets, all_predictions, args)
    primary_metric = select_primary_metric(test_metrics, args)

    return test_metrics, primary_metric


def select_primary_metric(metrics: dict, args: argparse.Namespace):
    ''' Select the primary metric based on task type. '''
    if args.task_type.lower() == 'binary':
        return metrics.get('auprc', -np.inf)
    
    elif args.task_type.lower() == 'multiclass':
        if args.key_detection:
            return metrics.get('weighted_accuracy', -np.inf)
        else:
            return metrics.get('top_1_accuracy', -np.inf)
        
    elif args.task_type.lower() == 'regression':
        dims = len([r for r in metrics if r.startswith('r2_dim')])
        if dims > 1:
            return np.mean([v for k, v in metrics.items() if k.startswith('r2_dim')])
        else:
            return metrics.get('r2', -np.inf)
    
    else:
        # no primary metric
        return -np.inf 
    

def save_embeds(train_dataset: EmbeddingDataset, valid_dataset: EmbeddingDataset, test_dataset: EmbeddingDataset, filepath: str):
    data = {
        'train': {
            'indices': train_dataset.indices.cpu(),
            'embeddings': train_dataset.embeddings.cpu(),
            'labels': train_dataset.labels
        },
        'valid': {
            'indices': valid_dataset.indices.cpu(),
            'embeddings': valid_dataset.embeddings.cpu(),
            'labels': valid_dataset.labels
        },
        'test': {
            'indices': test_dataset.indices.cpu(),
            'embeddings': test_dataset.embeddings.cpu(),
            'labels': test_dataset.labels
        }
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_embeds(filepath: str):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    train_dataset = EmbeddingDataset(
        indices=data['train']['indices'],
        embeddings=data['train']['embeddings'],
        labels=data['train']['labels']
    )

    valid_dataset = EmbeddingDataset(
        indices=data['valid']['indices'],
        embeddings=data['valid']['embeddings'],
        labels=data['valid']['labels']
    )

    test_dataset = EmbeddingDataset(
        indices=data['test']['indices'],
        embeddings=data['test']['embeddings'],
        labels=data['test']['labels']
    )

    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    args = parse_args()
    main(args)