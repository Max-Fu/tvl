# TVL-Encoder
## Training

To train ViT-Tiny (recommended) on the TVL dataset, please follow the dataset [setup](https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset). The top level folder should contain the two subsets, `ssvtp` and  `hct`. After obtaining the data path, please update `--datasets_dir` and run the following script:
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 main_pretrain.py --batch_size 256 --epochs 200 --warmup_epochs 10 --weight_decay 0.05 --datasets ssvtp hct --active_modality_names vision tactile text --find_unused_parameters --multi_epochs_dataloader --log_name tvl_vittiny_tactile_encoder --shuffle_text --no_text_prompt --replace_synonyms --num_workers 20 --use_not_contact --tactile_model vit_tiny_patch16_224 --blr 3e-4 --datasets_dir /your/data/dir
```
- To train ViT-Small or ViT-Base, set `--tactile_model vit_small_patch16_224` or `--tactile_model vit_base_patch16_224` and change base learning rate to `--blr 1.5e-4`. 
- To enable [FlashAttention2](https://github.com/Dao-AILab/flash-attention): please append the following flag `--enable_flash_attention2` to the command above 
- Training time is ~6.8h in 1 A100 GPUs (200 epochs) for ViT-Tiny and ViT-Small, and ~7.1h for ViT-Base. We find that usually IO is the bottleneck. An experimental method we use to speed this up is to move the entire dataset into `/dev/shm`. You can read more about it [here](https://www.cyberciti.biz/tips/what-is-devshm-and-its-practical-usage.html). 
- To use more than 1 GPU: update `CUDA_VISIBLE_DEVICES`, `--nproc_per_node` and `--batch_size` accordingly. This is only recommended for ViT-Base. 
- To perform tactile background subtraction, add `--subtract_background background`
- We do not see very significiant impact on performance when running at a higher learning rate. Please feel free to play around with the hyperparameters 
- `--use_not_contact` is the flag for using 10% background data. Turning it off will lead to a high validation loss; however, the performance on the test set seems to be roughly equal when an earlier checkpoint is used (i.e. `checkpoint-acc1.pth`).
- For legacy naming reasons, `tacvis.TacVisDataset` is the dataset object for SSVTP, `tacvis.TacVisDatasetV2` is the dataset object for HCT.

## Evaluation 
We use the following script to evaluate the model for touch-language classification on the test set:
```bash 
python -m tools.visualize_affinity --checkpoint_path output_dir/tvl_vittiny_tactile_encoder/checkpoint-acc1.pth --visualize_test --active_modality_names tactile text --tactile_model vit_tiny_patch16_224 --enable_flash_attention2 --no_text_prompt --datasets ssvtp hct --seed 42 --not_visualize --evaluate_all --similarity_thres 0.6356450319 0.8591097295 0.8927201033 0.9208499491 --datasets_dir /your/data/dir
```
And the following script to evaluate the model for touch-vision classification:
```bash 
python -m tools.visualize_affinity --checkpoint_path output_dir/tvl_vittiny_tactile_encoder/checkpoint-acc1.pth --visualize_test --active_modality_names tactile vision --tactile_model vit_tiny_patch16_224 --enable_flash_attention2 --no_text_prompt --datasets ssvtp hct --seed 42 --not_visualize --evaluate_all --datasets_dir /your/data/dir
```
To evaluate vision-language classification: 
```bash 
python -m tools.visualize_affinity --checkpoint_path output_dir/tvl_vittiny_tactile_encoder/checkpoint-acc1.pth --visualize_test --active_modality_names vision text --tactile_model vit_small_patch16_224 --enable_flash_attention2 --no_text_prompt --datasets ssvtp hct --seed 42 --not_visualize --evaluate_all --similarity_thres 0.6356450319 0.8591097295 0.8927201033 0.9208499491 --datasets_dir /your/data/dir
```
- To visualize the affinity matrix between the different modalities, you can remove the flag `--not_visualize` and add `--num_samples 16`. We find that visualization is quite noisy for touch-language and vision-language because the adjectives used to describe the surface are quite synonymous. 
- To see how the thresholds (`--similarity_thres`) are calculated, see [synonym_thres.py](synonym_thres.py). 

Once again, the checkpoints are provided below: 


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Tiny</th>
<th valign="bottom">ViT-Small</th>
<th valign="bottom">ViT-Base</th>
<!-- TABLE BODY -->
<tr><td align="left">Tactile Encoder</td>
<td align="center"><a href='https://huggingface.co/mlfu7/Touch-Vision-Language-Models/resolve/main/ckpt/tvl_enc/tvl_enc_vittiny.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/mlfu7/Touch-Vision-Language-Models/resolve/main/ckpt/tvl_enc/tvl_enc_vits.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/mlfu7/Touch-Vision-Language-Models/resolve/main/ckpt/tvl_enc/tvl_enc_vitb.pth?download=true'>download</a></td>
</tr>
<tr><td align="left">Touch-Language Acc (@0.64)</td>
<td align="center">36.19%</td>
<td align="center">36.82%</td>
<td align="center">30.85%</td>
</tr>
<tr><td align="left">Touch-Vision Acc</td>
<td align="center">78.11%</td>
<td align="center">77.49%</td>
<td align="center">81.22%</td>
</tr>
</tbody></table>