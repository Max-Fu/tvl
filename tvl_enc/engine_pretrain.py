# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import numpy as np

import torch

import util.misc as misc
import util.lr_sched as lr_sched

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def train_one_epoch(model: torch.nn.Module,
                    loss_fn: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples is a dictionary 
        for k, v in samples.items():
            if isinstance(v, list):
                v = v[0]
            samples[k] = v.to(device, non_blocking=True).squeeze()

        # loss, y, mask = model(samples, mask_ratio=mask_ratio, kept_mask_ratio=args.kept_mask_ratio)
        with torch.cuda.amp.autocast():
            out_dict = model(samples)
            loss_dict = loss_fn(out_dict, logit_scale=out_dict["logit_scale"], output_dict=True)

        # loss = loss_dict["average_loss"]
        loss = loss_dict.pop("average_loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        # log all keys in loss_dict
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        for k, v in loss_dict.items():
            loss_dict[k] = misc.all_reduce_mean(v.item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            # loss all keys in loss_dict
            for k, v in loss_dict.items():
                log_writer.add_scalar(f"train_{k}", v, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# TODO finish validation
@torch.no_grad()
def evaluate(data_loader, loss_fn, model, device, epoch=None, log_writer=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # samples is a dictionary 
        for k, v in batch.items():
            if isinstance(v, list):
                v = v[0]
            batch[k] = v.to(device, non_blocking=True).squeeze()
            batch_size = v.shape[0]

        # compute output
        with torch.cuda.amp.autocast():
            output = model(batch)
            loss_dict = loss_fn(output, logit_scale=output["logit_scale"], output_dict=True)

        loss = loss_dict.pop("average_loss")
        acc1, acc5 = loss_dict.pop("average_acc1"), loss_dict.pop("average_acc5")

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print("Averaged stats:", metric_logger)

    if log_writer is not None and epoch is not None: 
        for k, meter in metric_logger.meters.items():
            log_writer.add_scalar(f"val_{k}", meter.global_avg, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}