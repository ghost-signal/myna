# suppress warnings
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vit import SimpleViT # from vit_pytorch import SimpleViT
import wandb

from utils import *


def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    args.rank = rank
    args.world_size = world_size
    if world_size > 1:
        # Initialize process group for distributed training
        dist.init_process_group(backend=args.dist_backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        args.device = f'cuda:{rank}'
    else:
        # Use CPU or a single GPU
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, criterion, optimizer, train_loader, test_dataset, use_wandb, _ = setup_for_training(rank, world_size, args)

    if world_size > 1:
        # Wrap model in DDP
        model = DDP(model, device_ids=[rank], output_device=rank)

    best_test_metrics = {}
    for epoch in range(args.resume_epochs, args.epochs):
        if epoch == args.train_only_head_epochs:
            freeze_unfreeze_backbone(model, freeze=False)

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch, args=args)
        
        if rank == 0:
            test_loss, test_metrics = test(model, test_dataset, criterion, epoch, args=args)
            log_metrics(train_loss, test_loss, train_metrics, test_metrics, best_test_metrics, use_wandb)

        # checkpoint model
        if rank == 0 and args.checkpoint_dir and (epoch+1) % args.checkpoint_epochs == 0:
            save_model(model, args.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            save_optimizer(optimizer, criterion, args.checkpoint_dir, 'optimizer.pth')

    # log best values achieved
    if use_wandb:
        log_dict = {}
        for metric_name, value in best_test_metrics.items():
            log_dict[f'run_best/test_{metric_name}'] = value

        wandb.log(log_dict)

    # print best values
    for metric_name, value in best_test_metrics.items():
        print(f'Best {metric_name}: {value}')

    if world_size > 1:
        dist.destroy_process_group()

    print('Training complete.')
    return best_test_metrics


def setup_for_training(rank: int, world_size: int, args: argparse.Namespace):
    seed_everything(args.seed + rank)
    if rank == 0:
        print(f'==> Seed: {args.seed}')

    args.mel_frames = get_n_frames(args.n_samples, args)

    # initialize wandb
    use_wandb = False
    if rank == 0 and args.wandb:
        assert args.run_name is not None, 'wandb run needs to have a name.'
        assert args.wandb_project is not None, 'wandb project needs to be defined.'
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=args.run_name,
            reinit=False
        )
        use_wandb = True
        
    # load dataset
    train_dataset, train_loader = get_dataset(
        dataroot=os.path.join(args.dataroot, 'train'),
        args=args,
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size
    )
    test_dataset, _ = get_dataset(
        dataroot=os.path.join(args.dataroot, 'test'),
        args=args,
        drop_last=False
    )

    if rank == 0:
        print(f'==> Training dataset contains {len(train_dataset):,} songs.')
        print(f'==> Testing dataset contains {len(test_dataset):,} songs.')
        print(f'==> Using {args.device}')

    model = SimpleViT(
        image_size=(args.n_mels, args.mel_frames),
        channels=1,
        patch_size=args.patch_size,
        num_classes=args.num_outputs,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        additional_patch_size=args.additional_patch_size
    )

    if args.proj_head:
        add_proj_head(model, args.proj_head)

    if args.task_type.lower() == 'mae':
        setup_for_mae(model, args)

    if args.resume:
        load_model(model, args.resume, args.device, args.ignore_layers, verbose=(rank == 0))

    if args.use_other_patch_size:
        model.toggle_embeddings()

    if args.eval_hybrid:
        model.hybrid_mode = True
        args.dim *= 2

    # EXPERIMENTAL
    if args.add_layers:
        from vit_pytorch.simple_vit import Attention, FeedForward
        model.transformer.layers.extend([nn.ModuleList([
            Attention(dim=args.dim, heads=args.heads, dim_head=64),
            FeedForward(dim=args.dim, hidden_dim=args.mlp_dim)
        ]) for _ in range(args.add_layers)])

    if args.train_only_head_epochs:
        freeze_unfreeze_backbone(model, freeze=True)

    # EXPERIMENTAL
    if args.unfreeze_last_n_layers:
        for layer in model.transformer.layers[-args.unfreeze_last_n_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True

    model = model.to(args.device)

    criterion = get_criterion(args, N=len(train_dataset))
    optimizer = get_optimizer(model, args)

    if args.resume_optimizer:
        load_optimizer(optimizer, criterion, args.resume_optimizer)
        
    n_params = sum(p.numel() for p in model.parameters())
    n_active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active = f'({n_active:,} active)' if n_params != n_active else ''
    print(f'==> Model contains {n_params:,} parameters {active}')
    print(f'==> Using {args.optimizer.upper()} optimizer')

    return model, criterion, optimizer, train_loader, test_dataset, use_wandb, wandb


def log_metrics(train_loss: float, test_loss: float, train_metrics: dict, test_metrics: dict, best_test_metrics: dict, use_wandb: bool):
    # log to Weights and Biases
    if use_wandb:
        log_dict = {
            'train/loss': train_loss,
            'test/loss': test_loss
        }

        for metric_name, value in train_metrics.items():
            if metric_name == 'text': continue
            log_dict[f'train/{metric_name}'] = value
        for metric_name, value in test_metrics.items():
            if metric_name == 'text': continue
            log_dict[f'test/{metric_name}'] = value
            
        wandb.log(log_dict)

    for metric_name, value in test_metrics.items():
        if metric_name == 'text': continue
        if metric_name not in best_test_metrics or value > best_test_metrics[metric_name]:
            best_test_metrics[metric_name] = value


if __name__ == '__main__':
    args = parse_args()

    try:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    except KeyError:
        rank = 0
        world_size = 1

    main_worker(rank=rank, world_size=world_size, args=args)
