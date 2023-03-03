# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import os
import signal
import sys
import traceback
from pathlib import Path
from random import randint
import datetime

import torch
import wandb
import randomname
from pytorch_lightning.strategies import DDPStrategy

from util.distinct_colors import DistinctColors
from util.misc import visualize_depth, probability_to_normalized_entropy, get_boundary_mask
from util.warmup_scheduler import GradualWarmupScheduler
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from util.filesystem_logger import FilesystemLogger


def print_traceback_handler(sig, _frame):
    print(f'Received signal {sig}')
    bt = ''.join(traceback.format_stack())
    print(f'Requested stack trace:\n{bt}')


def quit_handler(sig, frame):
    print(f'Received signal {sig}, quitting.')
    sys.exit(1)


def register_debug_signal_handlers(sig=signal.SIGUSR1, handler=print_traceback_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def register_quit_signal_handlers(sig=signal.SIGUSR2, handler=quit_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def generate_experiment_name(name, config):
    if config.resume is not None:
        experiment = Path(config.resume).parents[1].name
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{datetime.datetime.now().strftime('%m%d%H%M')}_{name}_{config.experiment}_{randomname.get_name()}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


def create_trainer(name, config):
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = generate_experiment_name(name, config)
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)
    if isinstance(config.image_dim, int):
        config.image_dim = [config.image_dim, config.image_dim]
    assert config.image_dim[0] == config.image_dim[1], "only 1:1 supported"  # TODO: fix dataprocessing bug limiting this

    seed_everything(config.seed)

    register_debug_signal_handlers()
    register_quit_signal_handlers()

    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)

    # use wandb logger instead
    if config.logger == 'wandb':
        logger = WandbLogger(project=f'{name}{config.suffix}', name=config.experiment, id=config.experiment, settings=wandb.Settings(start_method='thread'))
    else:
        logger = TensorBoardLogger(name='tb', save_dir=(Path("runs") / config.experiment))

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment / "checkpoints"),
                                          save_top_k=-1,
                                          verbose=False,
                                          every_n_epochs=config.save_epoch)
    gpu_count = torch.cuda.device_count()

    if gpu_count > 1:
        # epoch_dependent_hparams_scale_factor = int(math.log2(gpu_count) + 1)
        # config.bbox_aabb_reset_epochs = [c * epoch_dependent_hparams_scale_factor for c in config.bbox_aabb_reset_epochs]
        # config.grid_upscale_epochs = [c * epoch_dependent_hparams_scale_factor for c in config.grid_upscale_epochs]
        trainer = Trainer(accelerator='gpu',
                          strategy=DDPStrategy(find_unused_parameters=True),
                          num_nodes=1,
                          devices=gpu_count,
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          resume_from_checkpoint=config.resume,
                          logger=logger,
                          benchmark=True)
    elif gpu_count == 1:
        trainer = Trainer(devices=[0],
                          accelerator='gpu',
                          strategy="ddp",
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          resume_from_checkpoint=config.resume,
                          logger=logger,
                          benchmark=True)
    else:
        trainer = Trainer(accelerator='cpu',
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          resume_from_checkpoint=config.resume,
                          logger=logger,
                          benchmark=True)
    return trainer


def step(opt, modules):
    for module in modules:
        for param in module.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    opt.step()


def get_optimizer_and_scheduler(params, config, betas=None):
    opt = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay, betas=betas)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=config.decay_step, gamma=config.decay_gamma)
    if config.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(opt, multiplier=config.warmup_multiplier, total_epoch=config.warmup_epochs, after_scheduler=scheduler)
    return opt, scheduler


def visualize_panoptic_outputs(p_rgb, p_semantics, p_instances, p_depth, rgb, semantics, instances, H, W, thing_classes, visualize_entropy=True):
    alpha = 0.65
    distinct_colors = DistinctColors()
    img = p_rgb.view(H, W, 3).cpu()
    img = torch.clamp(img, 0, 1).permute(2, 0, 1)
    if visualize_entropy:
        img_sem_entropy = visualize_depth(probability_to_normalized_entropy(torch.nn.functional.softmax(p_semantics, dim=-1)).reshape(H, W), minval=0.0, maxval=1.00, use_global_norm=True)
    else:
        img_sem_entropy = torch.zeros_like(img)
    if p_depth is not None:
        depth = visualize_depth(p_depth.view(H, W))
    else:
        depth = torch.zeros_like(img)
    if len(p_instances.shape) > 1 and len(p_semantics.shape) > 1:
        p_instances = p_instances.argmax(dim=1)
        p_semantics = p_semantics.argmax(dim=1)
    img_semantics = distinct_colors.apply_colors_fast_torch(p_semantics.cpu()).view(H, W, 3).permute(2, 0, 1) * alpha + img * (1 - alpha)
    boundaries_img_semantics = get_boundary_mask(p_semantics.cpu().view(H, W))
    img_semantics[:, boundaries_img_semantics > 0] = 0
    colored_img_instance = distinct_colors.apply_colors_fast_torch(p_instances.cpu()).float()
    boundaries_img_instances = get_boundary_mask(p_instances.cpu().view(H, W))
    colored_img_instance[boundaries_img_instances.reshape(-1) > 0, :] = 0
    thing_mask = torch.logical_not(sum(p_semantics == s for s in thing_classes).bool())
    colored_img_instance[thing_mask, :] = p_rgb.cpu()[thing_mask, :]
    img_instances = colored_img_instance.view(H, W, 3).permute(2, 0, 1) * alpha + img * (1 - alpha)
    if rgb is not None and semantics is not None and instances is not None:
        img_gt = rgb.view(H, W, 3).permute(2, 0, 1).cpu()
        img_semantics_gt = distinct_colors.apply_colors_fast_torch(semantics.cpu()).view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
        boundaries_img_semantics_gt = get_boundary_mask(semantics.cpu().view(H, W))
        img_semantics_gt[:, boundaries_img_semantics_gt > 0] = 0
        colored_img_instance_gt = distinct_colors.apply_colors_fast_torch(instances.cpu()).float()
        boundaries_img_instances_gt = get_boundary_mask(instances.cpu().view(H, W))
        colored_img_instance_gt[instances == 0, :] = rgb.cpu()[instances == 0, :]
        img_instances_gt = colored_img_instance_gt.view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
        img_instances_gt[:, boundaries_img_instances_gt > 0] = 0
        stack = torch.cat([torch.stack([img_gt, img_semantics_gt, img_instances_gt, torch.zeros_like(img_gt), torch.zeros_like(img_gt)]), torch.stack([img, img_semantics, img_instances, depth, img_sem_entropy])], dim=0)
    else:
        stack = torch.stack([img, img_semantics, img_instances, depth, img_sem_entropy])
    return stack
