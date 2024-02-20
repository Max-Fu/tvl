import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

from tvl_enc import tvl 
from tvl_enc.tvl import ModalityType
from loss import TVLLoss

from transformer_utils import handle_flash_attn

def print_losses(losses):
    # Determine the longest key length for alignment
    longest_key_length = max(len(key) for key in losses.keys())

    print("##### Loss and Accuracy Summary #####")
    print("-" * 35)
    for key, value in losses.items():
        # Calculate spaces needed for alignment
        spaces = longest_key_length - len(key)
        if "acc" in key:
            print(f"{' ' * spaces}{key}: {value.item():>7.2f}%")
        else:
            print(f"{' ' * spaces}{key}: {value.item():>7.4f}")
    print("-" * 35)
    print("#####################################")

def get_args_parser():
    parser = argparse.ArgumentParser('Tactile encoder visualization', add_help=False)
    parser.add_argument('--num_samples', default=32, type=int,
                        help='Number of samples to visualize.')

    # Dataset parameters
    parser.add_argument("--datasets_dir", type=str, default="./.datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--output_dir", type=str, default="vis_dir",
                        help="Directory to save the output")
    parser.add_argument("--datasets", type=str, default="ssvtp", nargs="+", choices=["ssvtp", "hct"],
                        help="Datasets to use for training and validation")
    parser.add_argument('--tactile_model', type=str, default='resnet18', choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224", "resnet18"], 
                        help="Tactile encoder model")
    parser.add_argument("--subtract_background", type=str, default=None, 
                        help="Subtract tactile by [mean, median] of the background tactile image", 
                        choices=[None, "mean", "median", "background"])
    parser.add_argument('--shuffle_text', action="store_true", help="Shuffle adjectives when feeding into the text modality")
    parser.add_argument('--no_text_prompt', action="store_true", help="Do not use text prompt when feeding into the text modality")
    parser.add_argument("--use_not_contact", action="store_true", default=False, help="Use not contact data (from tacvis v2 dataset)")
    parser.add_argument("--randomize_crop", action="store_true", default=False, help="Apply randomize_crop to the image modality, only available in tacvis v2 dataset")
    parser.add_argument("--similarity_thres", type=float, nargs="+", default=0.9, help="Similarity threshold for positive pairs")
    parser.add_argument('--common_latent_dim', type=int, default=None, help="Common latent dimension for all modalities, if is None, use open clip latent dimension")

    parser.add_argument("--visualize_train", action="store_true", help="Visualize train images.")
    parser.add_argument("--visualize_test", action="store_true", help="Visualize test images.")
    parser.add_argument("--not_visualize", action="store_true", help="Do not visualize images.")
    parser.add_argument("--evaluate_all", action="store_true", help="Evaluate all checkpoints in the output_dir.")

    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the model checkpoint.")
    parser.add_argument("--active_modality_names", nargs="+", type=str, default=["vision", "tactile"], 
                        choices=["vision", "text", "audio", "thermal", "depth", "imu", "tactile"],
                        help="Select 2 modalities to visualize")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--enable_flash_attention2', action='store_true', default=False, help="Use flash attntion 2")
    parser.add_argument("--color_jitter", action="store_true", default=False, help="Apply color jitter to the image modality")
    parser.add_argument("--use_old_statistics", action="store_true", default=False, help="use old statistics for tactile normalization")
    return parser

@torch.no_grad()
def main(args):
    assert len(args.active_modality_names) == 2, "Must select exactly 2 modalities to visualize affinity"
    assert args.visualize_train or args.visualize_test, "Please specify --visualize_train or --visualize_test"

    torch.backends.cudnn.determinstic = True

    if args.evaluate_all and args.visualize_train:
        raise ValueError("Cannot visualize all of train set when running --evaluate_all")

    import tacvis
    if args.use_old_statistics:
        tacvis.USE_OLD_STATISTICS = True
        tacvis.TAC_MEAN[:] = tacvis.TAC_MEAN_OLD[:]
        tacvis.TAC_STD[:] = tacvis.TAC_STD_OLD[:]

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True
    
    if args.enable_flash_attention2:
        handle_flash_attn(args)

    # handle the active modalities
    modality_types = []
    modalities = ["vision", "text", "tactile"]
    for modality_name in args.active_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            modality_types.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")
    modality_types = sorted(modality_types)

    # handle the datasets
    if args.no_text_prompt:
        prompt = ""
    else:
        prompt = "This image gives tactile feelings of "
    dataset = None
    datasets_eval = []
    print("datasets: ", args.datasets)
    if "ssvtp" in args.datasets:
        if args.subtract_background is None:
            tac_augments = tacvis.TAC_PREPROCESS
        elif args.subtract_background == "background":
            if args.color_jitter:
                tac_augments = tacvis.TAC_AUGMENTS_BG_CJ
            else:
                tac_augments = tacvis.TAC_AUGMENTS_BG
        else:
            raise ValueError(f"Unknown subtract_background: {args.subtract_background}")
        
        root_dir = os.path.join(args.datasets_dir, "ssvtp")
        if args.visualize_train:
            dataset = tacvis.TacVisDataset(
                root_dir=root_dir, split="train",
                transform_rgb=tacvis.RGB_AUGMENTS,
                transform_tac=tac_augments,
                modality_types=modality_types,
                shuffle_text=args.shuffle_text, 
                text_prompt=prompt,
                )
        else:
            dataset = tacvis.TacVisDataset(
                root_dir=root_dir, split="test",
                transform_rgb=tacvis.RGB_PREPROCESS,
                transform_tac=tac_augments,
                modality_types=modality_types,
                shuffle_text=False, 
                text_prompt=prompt,
                )
        datasets_eval.append(dataset)
    if "hct" in args.datasets:
        dataset_dirs = []
        for i in os.listdir(os.path.join(args.datasets_dir, "hct")):
            sub_dir = os.path.join(args.datasets_dir, "hct", i)
            if os.path.isdir(sub_dir) and os.path.exists(os.path.join(sub_dir, "contact.json")):
                dataset_dirs.append(sub_dir)
        print("dataset_dirs: ", dataset_dirs)
        if args.subtract_background is None:
            # we do not subtract background
            tac_augments = tacvis.TAC_PREPROCESS
        else:
            # we subtract background per calculated median background
            tac_augments = None
        if args.visualize_train:
            dataset = tacvis.TacVisDatasetV2(
                root_dir=dataset_dirs, split="train",
                transform_rgb=tacvis.RGB_AUGMENTS,
                modality_types=modality_types,
                randomize_crop=args.randomize_crop, 
                transform_tac=tac_augments,
                use_not_contact=args.use_not_contact,
                shuffle_text=False,
                text_prompt=prompt, 
            )
        else:
            dataset = tacvis.TacVisDatasetV2(
                root_dir=dataset_dirs, split="test",
                transform_rgb=tacvis.RGB_PREPROCESS,
                modality_types=modality_types,
                transform_tac=tac_augments,
                use_not_contact=False,
                shuffle_text=False,
                text_prompt=prompt, 
            )
        datasets_eval.append(dataset)
    
    if len(datasets_eval) == 1:
        dataset = datasets_eval[0]
    else:
        dataset = ConcatDataset(datasets_eval)

    sampler = torch.utils.data.RandomSampler(dataset)

    if args.evaluate_all:
        num_samples = len(dataset)
        print(f"using all of test set, which contains {num_samples} images")
    else:
        num_samples = args.num_samples

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=num_samples,
    )

    model = tvl.TVL(active_modalities=modality_types, tactile_model=args.tactile_model, common_latent_dim=args.common_latent_dim)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if args.common_latent_dim is None:
        assert len(unexpected_keys) == 0, f"Unexpected keys found in the checkpoint: {unexpected_keys}"
    else:
        assert len(unexpected_keys) <= 2, f"Unexpected keys found in the checkpoint: {unexpected_keys}"
    model.eval()

    samples = next(iter(data_loader))
    for k, v in samples.items():
        if isinstance(v, list):
            v = v[0]
        samples[k] = v.to(device, non_blocking=True).squeeze()

    with torch.cuda.amp.autocast():
        out_dict = model(samples)
        affinity_matrix = out_dict[modality_types[1]] @ out_dict[modality_types[0]].T
    
        print("modality_types: ", modality_types)
        if not isinstance(args.similarity_thres, list):
            args.similarity_thres = [args.similarity_thres]
        for st in args.similarity_thres:
            loss_fn = TVLLoss(active_modalities=modality_types, similarity_thres=st)
            loss_dict = loss_fn(out_dict, logit_scale=out_dict["logit_scale"], output_dict=True)
            print(f"Evaluating at Similarity Threshold: {st}")
            print_losses(loss_dict)
    
    # save loss_dict, with name based on the args.checkpoint_path 
    checkpoint_name = args.checkpoint_path.split("/")[-2]
    args.output_dir = os.path.join(args.output_dir, checkpoint_name)
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"{checkpoint_name}_loss_dict.json")

    # cast all loss_dict into numbers 
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            loss_dict[k] = v.item()

    with open(out_json, "w") as f:
        json.dump(loss_dict, f)

    if ModalityType.TEXT in modality_types:
        text_feats = out_dict[ModalityType.TEXT]
        text_am = text_feats @ text_feats.T
        # plot text_am 
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(text_am.cpu().numpy(), cmap='viridis')
        fig.colorbar(cax)
        image_path = os.path.join(args.output_dir, f"{checkpoint_name}_text_am.png")
        plt.savefig(image_path)
        plt.clf()
        
        for st in args.similarity_thres:
            text_am_mask = text_am > st
            # plot text_am_mask
            fig, ax = plt.subplots(figsize=(10, 10))
            cax = ax.matshow(text_am_mask.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax)
            image_path = os.path.join(args.output_dir, f"{checkpoint_name}_text_am_mask_{st}.png")
            plt.savefig(image_path)
            plt.clf()

    print("Affinities", affinity_matrix)
    
    if args.not_visualize:
        exit()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Display affinity matrix
    cax = ax.matshow(affinity_matrix.cpu().numpy(), cmap='viridis')
    fig.colorbar(cax)

    # if more than 32 images it is very hard to visualize
    if torch.any(torch.tensor(affinity_matrix.shape) <= 32):
        if ModalityType.VISION in samples:
            unnorm_vision = tacvis.unnormalize_fn(tacvis.RGB_MEAN, tacvis.RGB_STD)
            samples[ModalityType.VISION] = [unnorm_vision(img) for img in samples[ModalityType.VISION]]
        if ModalityType.TACTILE in samples:
            if args.subtract_background == "mean": 
                unnorm_tactile = tacvis.TAC_AVG_UNDO
            elif args.subtract_background == "median":
                unnorm_tactile = tacvis.TAC_MEDIAN_UNDO
            elif args.subtract_background is None:
                unnorm_tactile = tacvis.unnormalize_fn(tacvis.TAC_MEAN, tacvis.TAC_STD)
            elif args.subtract_background == "background":
                unnorm_tactile = tacvis.TAC_BG_UNDO
            samples[ModalityType.TACTILE] = [unnorm_tactile(tac) for tac in samples[ModalityType.TACTILE]]

        if ModalityType.TEXT in samples:
            text_data = samples[ModalityType.TEXT]
            last_idxs = [torch.nonzero(t)[-1].cpu().item() for t in text_data]
            samples[ModalityType.TEXT] = [tvl.tokenizer.decode(t[1:last_idxs[i]].cpu().numpy()) for i, t in enumerate(text_data)]

        for i, data in enumerate(samples[modality_types[0]]):
            inset_axes = ax.inset_axes([i/args.num_samples, -0.2, 1/args.num_samples, 0.2])
            if modality_types[0] == ModalityType.TEXT:
                inset_axes.text(0, 0.45, data)
            else:
                inset_axes.imshow(data)
            inset_axes.axis('off')

        for i, data in enumerate(reversed(samples[modality_types[1]])):
            inset_axes = ax.inset_axes([-0.2, i/args.num_samples, 0.2, 1/args.num_samples])
            if modality_types[1] == ModalityType.TEXT:
                inset_axes.text(0, 0.45, data)
            else:
                inset_axes.imshow(data)
            inset_axes.axis('off')

    image_path = os.path.join(args.output_dir, f"{checkpoint_name}_affinity.png")
    plt.savefig(image_path)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

