import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import os
from tvl_enc import tacvis

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=tacvis.RGB_MEAN, std=tacvis.RGB_STD)])

# based on RGB_AUGMENTS in tacvis.py
transform_train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=(0.9, 1.1),
        contrast=(.9, 1.1),
        saturation=0.1,
    )], p=.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(9, sigma=(.5, 1))], p=.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=tacvis.RGB_MEAN, std=tacvis.RGB_STD)])

class FinetuneDataset(Dataset):
    def __init__(
        self, config_path, max_words=30, tokenizer_path=None, 
        crop_tacvis=False, subtract_background=None, 
        augment_rgb=False, augment_tactile=False,
        random_drop=False, 
    ):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            folder_path = os.path.dirname(meta_path)
            meta_l = json.load(open(meta_path))
            for item in meta_l:
                item["folder_path"] = folder_path
            print(f"{meta_path}: len {len(meta_l)}")
            ann += meta_l
        self.ann = ann
        print(f"total length: {len(self)}")
        self.transform = transform_train_aug if augment_rgb else transform_train
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.crop_tacvis = crop_tacvis
        print("enable crop_tacvis: ", crop_tacvis)
        print("subtract background: ", subtract_background)
        self.subtract_background = subtract_background
        self.random_drop = random_drop 
        print("random drop: ", random_drop)
        if subtract_background is None:
            self.tactile_process = tacvis.TAC_AUGMENTS if augment_tactile else tacvis.TAC_WBG
        elif subtract_background == "background": 
            self.tactile_process = tacvis.TAC_AUGMENTS_BG if augment_tactile else tacvis.TAC_BG

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item.keys():
            filename = data_item['image']
            tac_filename = data_item.get("tactile", None)
            tac_background_filename = data_item.get("tactile_background", None)
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']
            folder_path = data_item['folder_path']
            if "ssvtp" in folder_path or "hct" in folder_path:
                filename = os.path.join(folder_path, filename)
                tac_filename = os.path.join(folder_path, tac_filename)
                if tac_background_filename is not None:
                    tac_background_filename = os.path.join(folder_path, tac_background_filename)
                if self.crop_tacvis:
                    if "ssvtp" in folder_path:
                        image = tacvis.load_vision_data(filename)
                    elif "hct" in folder_path:
                        image = tacvis.load_vision_data(filename, dataset_version="v2", randomize_crop=True)
                else:
                    image = Image.open(filename).convert('RGB')
                    image = self.transform(image)
                if "ssvtp" in folder_path:
                    tactile = tacvis.load_tactile_data(tac_filename, transform_tac=self.tactile_process)
                elif "hct" in folder_path:
                    if self.subtract_background is None:
                        tactile = tacvis.load_tactile_data(tac_filename, transform_tac=self.tactile_process)
                    else:
                        transform_tac = tacvis.tac_subtract_bg(tac_background_filename, tacvis.TAC_MEAN_BG, tacvis.TAC_STD_BG)
                        tactile = tacvis.load_tactile_data(tac_filename, transform_tac=transform_tac)
                
                # process random drop logic: can drop at most one modality
                if self.random_drop:
                    drop = random.choice([0, 1, 2]) # for 2, both modality are kept
                    if drop == 0:
                        image = torch.zeros(3, 224, 224)
                    elif drop == 1:
                        tactile = torch.zeros(3, 224, 224)

            else:
                image = Image.open(filename).convert('RGB')
                image = self.transform(image)
                tactile = torch.zeros(3, 224, 224)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 224, 224)
            tactile = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return {
            "input2" :input2, 
            "labels" : labels,
            "input2_mask" : input2_mask,
            "image" : image,
            "tactile" : tactile, 
        }


class PretrainDataset(Dataset):
    def __init__(
        self, config_path, max_words=30, 
        tokenizer_path=None, crop_tacvis=False, subtract_background=None, 
        augment_rgb=False, augment_tactile=False, random_drop=False
    ):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        augs, images, captions, tactiles, tactile_backgrounds = [], [], [], [], []
        for meta_path in self.config['META']:
            images_this_meta, tactiles_this_meta, tactile_backgrounds_this_meta, captions_this_meta = [], [], [], []
            folder_path = os.path.dirname(meta_path)
            for chunk in pd.read_csv(meta_path, chunksize=10 ** 6):
                imgs = chunk['url'].tolist()
                imgs = [os.path.join(folder_path, img) for img in imgs]
                images_this_meta.extend(imgs)
                if "ssvtp" in meta_path:
                    tactiles_this_meta.extend([os.path.join(folder_path, t) for t in chunk['tactile'].tolist()])
                    tactile_backgrounds_this_meta.extend([None] * len(chunk))
                elif "hct" in meta_path:
                    tactiles_this_meta.extend([os.path.join(folder_path, tactile) for tactile in chunk['tactile'].tolist()])
                    tactile_backgrounds_this_meta.extend(
                        [os.path.join(folder_path, tb) for tb in chunk['tactile_background'].tolist()]
                    )
                else:
                    tactiles_this_meta.extend([None] * len(chunk))
                    tactile_backgrounds_this_meta.extend([None] * len(chunk))
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            if "ssvtp" in meta_path:
                augs.extend(["ssvtp"] * len(images_this_meta))
            elif "hct" in meta_path:
                augs.extend(["hct"] * len(images_this_meta))
            else:
                augs.extend(["clip"] * len(images_this_meta))
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)
            tactiles.extend(tactiles_this_meta)
            tactile_backgrounds.extend(tactile_backgrounds_this_meta)

        self.data_list = []
        for a, x, y, t, bg in zip(augs, images, captions, tactiles, tactile_backgrounds):
            self.data_list.append({'aug': a, 'url': x, 'caption': y, 'tactile': t, "tactile_background": bg})
        print(f"total length: {len(self)}")
        print("enable rgb augmentation: ", augment_rgb)
        self.transform = transform_train_aug if augment_rgb else transform_train
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.crop_tacvis = crop_tacvis
        self.random_drop = random_drop
        print("enable crop_tacvis: ", crop_tacvis)
        print("subtract background: ", subtract_background)
        print("augment tactile: ", augment_tactile)
        print("random drop: ", random_drop)
        self.subtract_background = subtract_background
        if subtract_background is None:
            self.tactile_process = tacvis.TAC_AUGMENTS if augment_tactile else tacvis.TAC_WBG
        elif subtract_background == "background": 
            self.tactile_process = tacvis.TAC_AUGMENTS_BG if augment_tactile else tacvis.TAC_BG

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        when tactile data is not present in the dataset, tactile will be None
        """
        sample = self.data_list[index]
        aug, image_path, caption, tactile, tactile_background = sample['aug'], sample['url'], sample['caption'], sample['tactile'], sample["tactile_background"]
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)
        if aug == "ssvtp" or aug == "hct":
            if self.crop_tacvis:
                if aug == "hct":
                    image = tacvis.load_vision_data(image_path, dataset_version="v2", randomize_crop=True)
                else:
                    image = tacvis.load_vision_data(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            
            if aug == "ssvtp":
                tactile = tacvis.load_tactile_data(tactile, transform_tac=self.tactile_process)
            elif aug == "hct":
                if self.subtract_background is None:
                    tactile = tacvis.load_tactile_data(tactile, transform_tac=self.tactile_process)
                else:
                    transform_tac = tacvis.tac_subtract_bg(tactile_background, tacvis.TAC_MEAN_BG, tacvis.TAC_STD_BG)
                    tactile = tacvis.load_tactile_data(tactile, transform_tac=transform_tac)

            # process random drop logic: can drop at most one modality
            if self.random_drop:
                drop = random.choice([0, 1, 2]) # for 2, both modality are kept
                if drop == 0:
                    image = torch.zeros(3, 224, 224)
                elif drop == 1:
                    tactile = torch.zeros(3, 224, 224)
            format_instruction = "This image gives tactile feelings of"

        elif aug == "clip":
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            tactile = torch.zeros(3, 224, 224)
            format_instruction = "Generate caption of this image"

        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return {
            "input2" :input2, 
            "labels" : labels,
            "input2_mask" : input2_mask,
            "image" : image,
            "tactile" : tactile, 
        }
