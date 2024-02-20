import os
import os.path as osp
import json
from typing import Optional, Callable
from itertools import chain, combinations

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import re
from typing import Union
from copy import deepcopy

from tvl_enc.tvl import ModalityType, tokenizer
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image 
import torchvision.transforms.functional as TF
import numpy as np
import random
import csv 
import pathlib

RGB_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
RGB_STD = np.array([0.26862954, 0.26130258, 0.27577711])

TAC_MEAN = np.array([0.29174602047139075, 0.2971325588927249, 0.2910404549605639])
TAC_STD = np.array([0.18764469044810236, 0.19467651810273057, 0.21871583397361818])

data_path = pathlib.Path(__file__).resolve().parent
TAC_BG_FP = f"{data_path}/data/tac_background.png"
TAC_MEAN_BG = np.array([-0.00809318389762342, -0.01887447008747725, -0.018430588238856332])
TAC_STD_BG = np.array([0.04535400223885517, 0.044029170444552575, 0.05332520729596308])

BPE_PATH = "bpe/bpe_simple_vocab_16e6.txt.gz"

def to_pil(img : torch.Tensor):
    img = np.moveaxis(img.numpy()*255, 0, -1)
    return Image.fromarray(img.astype(np.uint8))

def unnormalize_fn(mean : tuple, std : tuple) -> transforms.Compose:
    """
    returns a transformation that turns torch tensor to PIL Image
    """
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            ),
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)), 
            transforms.ToPILImage(),
        ]
    )

TO_TENSOR = transforms.Compose([
    transforms.ToTensor()
])

# this describes augments for training which jitter color, similar to ImageBind
# the key difference is that we don't do random resize and crop
RGB_AUGMENTS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=(0.8, 1.1),
        contrast=(.7, 1.3),
        saturation=0.2,
        hue=0.0
    )], p=.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(9, sigma=(.5, 1))], p=.5),
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=RGB_MEAN,
        std=RGB_STD,
    ),
])

RGB_PREPROCESS = transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=RGB_MEAN,
        std=RGB_STD,
    ),
])

class RandomDiscreteRotation(nn.Module):
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

# in the original tactile vision paper, gaussian blur is not applied
TAC_AUGMENTS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=(0.9, 1.1),
        contrast=(.9, 1.1),
        saturation=0.2,
        hue=0.05
    )], p=.8),
    # randomly apply exactly 90 degree rotation
    RandomDiscreteRotation([0, 90]),
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TAC_MEAN,
        std=TAC_STD,
    ),
])

TAC_PREPROCESS = transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TAC_MEAN,
        std=TAC_STD,
    ),
])

# preprocess with background
TAC_WBG = transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TAC_MEAN,
        std=TAC_STD,
    ),
])


def tac_padding(
    tac : Union[torch.Tensor, Image.Image], 
): 
    if isinstance(tac, Image.Image):
        tac_w, tac_h = tac.size
    else:
        tac_h, tac_w = tac.shape[1:]
    hpad = int(np.clip(max(tac_h, tac_w) - tac_h, 0, np.inf) / 2)
    wpad = int(np.clip(max(tac_h, tac_w) - tac_w, 0, np.inf) / 2)
    tac = TF.pad(tac, [wpad, hpad])
    tac = TF.rotate(tac, 90)
    return tac

class BackgroundOps(torch.nn.Module):
    def __init__(self, background_fp : str, op : str = "subtract", padding : bool = True) -> None:
        """
        tac padding is necessary for training
        """
        super().__init__()
        self.background_fp = background_fp
        self.background = Image.open(background_fp)
        self.background = TO_TENSOR(self.background)
        if padding: 
            self.background = tac_padding(self.background)
        assert op in ["subtract", "add"], "op must be either subtract or add"
        if op == "subtract":
            self.background = -self.background
        
    def forward(self, img : Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Args: 
            img: either a PIL Image or a torch Tensor
        Returns:
            a torch Tensor
        """
        if isinstance(img, Image.Image):
            img = TO_TENSOR(img)
        return img + self.background.to(img.device)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(background={self.background_fp})"

class SyncRandomBackgroundSubtract(torch.nn.Module):
    def __init__(self, transform : torch.nn.Module, background_fp : str, p : float = 0.8) -> None:
        super().__init__()
        # create background op 
        self.background_fp = background_fp
        # adding padding, as images are padded beforehand
        self.background = Image.open(background_fp)
        self.background = TO_TENSOR(self.background)
        self.background = tac_padding(self.background)
        self.background = to_pil(self.background)
        self.p = p # probability of applying the transform
        self.transform = transform 
        assert isinstance(self.transform, transforms.ColorJitter), "transform must be a ColorJitter transform"

    def forward(self, img : Image.Image) -> torch.Tensor:
        if np.random.rand() < self.p:
            _, brightness, contrast, saturation, hue = self.transform.get_params(
                self.transform.brightness, self.transform.contrast, self.transform.saturation, self.transform.hue)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.adjust_hue(img, hue)

            background = TF.adjust_brightness(deepcopy(self.background), brightness)
            background = TF.adjust_contrast(background, contrast)
            background = TF.adjust_saturation(background, saturation)
            background = TF.adjust_hue(background, hue)
        else:
            background = deepcopy(self.background)
        img, background = TO_TENSOR(img), TO_TENSOR(background)

        subtracted = img - background
        return subtracted
    
    def __repr__(self):
        return f"{self.__class__.__name__}(background={self.background_fp}, transform={self.transform})"

# Tactile image augmentation that subtract the background with filepath, new mean and std, using synchronized colorjitters
def tac_subtract_bg_sync_aug(fp : str, mean : tuple, std : tuple, p : float = 0.8) -> transforms.Compose:
    cj = transforms.ColorJitter(
        brightness=(0.9, 1.1),
        contrast=(.9, 1.1),
        saturation=0.2,
        hue=0.05
    )
    all_trs = [
        SyncRandomBackgroundSubtract(transform=cj, background_fp=fp, p=p),
        transforms.RandomHorizontalFlip(),
        RandomDiscreteRotation([0, 90]),
        transforms.Resize(size=224),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ]
    return transforms.Compose(all_trs)

# Tactile image augmentation that subtract the background with filepath, new mean and std
def tac_subtract_bg_aug(fp : str, mean : tuple, std : tuple, color_jitter : bool = True) -> transforms.Compose:
    all_trs = [
        BackgroundOps(fp, op="subtract"), # This does to tensor already
        transforms.RandomHorizontalFlip(),
        RandomDiscreteRotation([0, 90]),
        transforms.Resize(size=224),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ]
    if color_jitter:
        cj = transforms.RandomApply([transforms.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(.9, 1.1),
            saturation=0.2,
            hue=0.05
        )], p=.8)
        all_trs = [cj] + all_trs
    return transforms.Compose(all_trs)

# Tactile image preprocessing that subtract the background with filepath, new mean and std
tac_subtract_bg = lambda fp, mean, std : transforms.Compose([
    BackgroundOps(fp, op="subtract"),
    transforms.Resize(size=224),
    transforms.Normalize(
        mean=mean,
        std=std,
    ),
])

# Tactile image augmentation that subtract the background with true background tactile image
TAC_AUGMENTS_BG = tac_subtract_bg_aug(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG, color_jitter=False)
TAC_AUGMENTS_BG_CJ = tac_subtract_bg_sync_aug(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG)
TAC_BG = tac_subtract_bg(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG)

def unnormalize_fn_bg(fp : str, mean : tuple, std : tuple) -> transforms.Compose:
    """
    undo background subtraction and normalization
    returns a transformation that turns torch tensor to PIL Image
    """
    background_ops = BackgroundOps(fp, op="add")
    h, w = background_ops.background.shape[1:]
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            ),
            transforms.Resize(size=(h, w)),
            background_ops,
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)), 
            transforms.ToPILImage(),
        ]
    )

# The inverse function of TAC_BG
TAC_BG_UNDO = unnormalize_fn_bg(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG)

def load_vision_data(
    path : str, 
    rgb_size : list = [224, 224], 
    im_scale_range : list = [.12, .18], 
    transform_rgb = transforms.Compose([
        transforms.Resize(size=224), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=RGB_MEAN,
            std=RGB_STD,
        ),
    ]), 
    dataset_version : str = "v1",
    randomize_crop : bool = False, # only supported in v2
    randomize_range : float = 0.05, # only supported in v2
    device : str = None,
):
    assert dataset_version in ["v1", "v2"]
    rgb = Image.open(path)
    rgb_w, rgb_h = rgb.size
    
    if dataset_version == "v1":
        rgb = TF.center_crop(rgb, np.ceil(
            np.sqrt(2) * im_scale_range[1] * max(rgb_h, rgb_w)))
        
    elif dataset_version == "v2":
        # Calculate the size for cropping
        crop_height = int(np.ceil(np.sqrt(2) * im_scale_range[1] * max(rgb_h, rgb_w)))
        crop_width = crop_height  # Width remains the same

        # Calculate top left corner of the crop
        if randomize_crop:
            top_random_range = int(randomize_range * crop_height)
            left_random_range = int(randomize_range * crop_width)
            # Crop from the top, for data3 images, consistent with gpt psuedo label generation
            start_pos = 0 if "data3" not in path else 200  
            top = np.random.randint(start_pos, top_random_range + start_pos)
            left = np.random.randint(-left_random_range // 2, left_random_range // 2) + (rgb_w - crop_width) // 2
        else:
            # Crop from the top, for data3 images, consistent with gpt psuedo label generation
            top = 0 if "data3" not in path else 200 
            left = (rgb_w - crop_width) // 2  # Center horizontally
        rgb = TF.crop(rgb, top, left, crop_height, crop_width)
    # rgb = to_pil(rgb)
    if transform_rgb is not None: 
        rgb = transform_rgb(rgb)
    if device is not None:
        rgb = rgb.to(device)
    return rgb

def load_tactile_data(
    path : str, 
    transform_tac = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TAC_MEAN,
            std=TAC_STD,
        ),
    ]),
    device : str = None,
):
    tac = Image.open(path)
    # tac = TO_TENSOR(tac)
    tac = tac_padding(tac)
    # tac = to_pil(tac)
    if transform_tac is not None:
        tac = transform_tac(tac)
    if device is not None:
        tac = tac.to(device)
    return tac

def load_text(
    raw_text: str,
    prompt: str = "This image gives tactile feelings of ",
    device: str = None,
    shuffle: bool = False, 
    random_subset: bool = True,
    synonyms_dict: dict = None
):
    keywords = raw_text.replace(".", "").replace("\n", "").lower().split(',')
    keywords = [k.strip() for k in keywords]
    if random_subset:
        ps = chain.from_iterable(combinations(keywords, r) for r in range(len(keywords)+1))
        ps = list(ps)[1:] # drop first element (the empty set)
        text = prompt
        words = ps[np.random.choice(len(ps))]
        # words = [np.random.choice(keywords)]
        if shuffle:
            words = list(words)
            random.shuffle(words)
        if synonyms_dict is not None:
            selected = []
            for w in words:
                if w in synonyms_dict:
                    selected.append(np.random.choice(synonyms_dict[w]))
                else:
                    selected.append(w)
            words = selected
    else:
        text = prompt
        words = keywords
    if len(words) == 1:
        text += words[0]
    elif len(words) == 1:
        text += f"{words[0]} and {words[1]}"
    else:
        text += ", ".join(words[i] for i in range(len(words) - 1)) + f", and {words[-1]}"
    text += "."
    tokens = tokenizer(text)
    if device is not None:
        tokens = tokens.to(device)
    return tokens

def load_text_data(
    path: str,
    prompt: str = "This image gives tactile feelings of ",
    device: str = None,
    shuffle: bool = False, 
    random_subset: bool = True,
    synonyms_dict: dict = None,
):
    with open(path, 'r') as f:
        raw_text = f.readline()
    return load_text(raw_text, prompt, device, shuffle, random_subset, synonyms_dict)

class TacVisDatasetV2(Dataset):
    def __init__(self, root_dir: str, transform_rgb: Optional[Callable] = None, transform_tac: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.9, random_seed: int = 42, 
                 modality_types = [ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT],
                 device: str = 'cpu', rgb_size=[224, 224], tac_size=[224, 224], im_scale_range=[.12, .18], randomize_crop=False,
                 use_not_contact=False, shuffle_text : bool = False, text_prompt="This image gives tactile feelings of ", 
                 replace_synonyms=False, keep_k_synonyms=None, percent_not_contact=0.1,
                 ):
        self.rgb_size = rgb_size
        self.tac_size = tac_size
        self.im_scale_range = im_scale_range
        self.randomize_crop = randomize_crop
        self.replace_synonyms = replace_synonyms # TODO set this up for tacvis-v2 
        self.keep_k_synonyms = keep_k_synonyms # TODO set this up for tacvis-v2 
        self.text_prompt = text_prompt 
        self.shuffle_text = shuffle_text
        print("data_dir: ", root_dir)
        assert split in ["train", "val", "test"], "split must be either train, val, or test"
        print("split: ", split)
        print("shuffling text: ", self.shuffle_text)
        print("text prompt: ", self.text_prompt)

        if not isinstance(root_dir, list):
            self.root_dir = [root_dir]
        else:
            self.root_dir = root_dir
        self.transform_rgb = transform_rgb
        self.transform_tac = transform_tac
        self.device = device
        self.modality_types = modality_types

        self.data = {"tactile": [], "vision": [], "background": [], "text": []}
        for data_dir in self.root_dir:
            if split in ["train", "val"]:
                split_csv = os.path.join(data_dir, "train.csv")
            else:
                split_csv = os.path.join(data_dir, "test.csv")
            if os.path.exists(split_csv):
                print("using ", split_csv)
                with open(split_csv, 'r') as f:
                    reader = csv.reader(f)
                    rows = [row for row in reader][1:]
                    for row in rows:
                        vision, tactile, background, caption = row
                        self.data["vision"].append(os.path.join(data_dir, vision))
                        self.data["tactile"].append(os.path.join(data_dir, tactile))
                        self.data["background"].append(os.path.join(data_dir, background))
                        self.data["text"].append(caption) # can be added later
            else:
                data_json = os.path.join(data_dir, "contact.json")
                with open(data_json, "r") as f:
                    data = json.load(f)
                self.data["text"].extend([None] * len(data["vision"]))
                for k in data:
                    self.data[k].extend([osp.join(data_dir, e) for e in data[k]])

        if use_not_contact:
            print("using not contact data, percentage of data does not exceed: ", percent_not_contact)
            self.not_in_contact = {"tactile": [], "vision": [], "background": [], "text": []}
            for data_dir in self.root_dir:
                data_json = os.path.join(data_dir, "not_contact.json")
                with open(data_json, "r") as f:
                    data = json.load(f)
                for k in data:
                    self.not_in_contact[k].extend([osp.join(data_dir, e) for e in data[k]])
                self.not_in_contact["text"].extend(["background"] * len(data["vision"]))
            if len(self.data["vision"]) * percent_not_contact < len(self.not_in_contact["vision"]):
                indices = np.random.choice(len(self.not_in_contact["vision"]), int(len(self.data["vision"]) * percent_not_contact), replace=False)
                self.not_in_contact = {k: [self.not_in_contact[k][i] for i in indices] for k in self.not_in_contact}
            # merge the two datasets
            for k in self.not_in_contact:
                self.data[k].extend(self.not_in_contact[k])

        # Split dataset
        if split in ["train", "val"]:
            indices = np.arange(len(self.data["vision"]))
            train_indices, test_indices = train_test_split(indices, train_size=train_size, random_state=random_seed)
        if split == 'train':
            print("number of training samples: ", len(train_indices))
        elif split == 'val':
            print("number of testing samples: ", len(test_indices))
        elif split == "test":
            print("number of testing samples: ", len(self.data["vision"]))
        if split == 'train':
            self.paths = {k: [self.data[k][i] for i in train_indices] for k in self.data}
        elif split == 'val':
            self.paths = {k: [self.data[k][i] for i in test_indices] for k in self.data}
        elif split == "test":
            self.paths = self.data

    def __repr__(self):
        return f"{self.__class__.__name__}(root_dir={self.root_dir}, split={self.split}, modality_types={self.modality_types})"

    def __len__(self):
        return len(self.paths["vision"])
    
    def load_vision_data(self, path : str):
        return load_vision_data(
            path, self.rgb_size, self.im_scale_range, transform_rgb=self.transform_rgb, 
            dataset_version="v2", randomize_crop=self.randomize_crop
        )
    
    def load_tactile_data(self, path: str, background_path: str): 
        if self.transform_tac is None:
            # by default, we perform background subtraction per trajectory
            if not os.path.exists(background_path):
                background_path = os.path.join("/".join(background_path.split("/")[:-3]), "digit_background.png")
            transform_tac = tac_subtract_bg_aug(background_path, TAC_MEAN_BG, TAC_STD_BG, color_jitter=False)
            return load_tactile_data(path, transform_tac=transform_tac)
        else:
            return load_tactile_data(path, transform_tac=self.transform_tac)

    def load_text_data(self, index : int):
        """
        Behavior different from V1: here we directly get the index from self.data
        using a placeholder so that it can be easily filtered
        """
        if self.paths["text"][index] is not None:
            return load_text(self.paths["text"][index], prompt=self.text_prompt, shuffle=self.shuffle_text)
        else:
            return torch.zeros(1, 77, dtype=torch.long)
    
    def __getitem__(self, index):
        item = OrderedDict()
        if ModalityType.VISION in self.modality_types:
            images = self.load_vision_data(self.paths["vision"][index])
            item[ModalityType.VISION] = [images]
        if ModalityType.TACTILE in self.modality_types:
            tactiles = self.load_tactile_data(self.paths["tactile"][index], self.paths["background"][index])
            item[ModalityType.TACTILE] = [tactiles]
        if ModalityType.TEXT in self.modality_types:
            texts = self.load_text_data(index)
            item[ModalityType.TEXT] = texts
        return item

class TacVisDataset(Dataset):
    def __init__(self, root_dir: str, transform_rgb: Optional[Callable] = None, transform_tac: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.9, random_seed: int = 42, 
                 modality_types = [ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT],
                 device: str = 'cpu', rgb_size=[224, 224], tac_size=[224, 224], im_scale_range=[.12, .18], 
                 shuffle_text : bool = False, text_prompt="This image gives tactile feelings of ", replace_synonyms=False, keep_k_synonyms=None,):
        # assert ModalityType.TEXT not in modality_types, "currently do not support TEXT"
        self.rgb_size = rgb_size
        self.tac_size = tac_size
        self.im_scale_range = im_scale_range

        if not isinstance(root_dir, list):
            self.root_dir = [root_dir]
        self.transform_rgb = transform_rgb
        self.transform_tac = transform_tac
        self.device = device
        self.modality_types = modality_types
        self.shuffle_text = shuffle_text
        self.text_prompt = text_prompt
        self.replace_synonyms = replace_synonyms
        print("data_dir: ", root_dir)
        print("split: ", split)
        print("shuffling text: ", self.shuffle_text)
        print("text prompt: ", self.text_prompt)

        self.paths = []
        self.synonyms = {}
        for data_dir in self.root_dir:
            if split in ["train", "val"]:
                data_dir_csv = os.path.join(data_dir, "train.csv")
            else:
                data_dir_csv = os.path.join(data_dir, "test.csv")
            if os.path.exists(data_dir_csv):
                print("loading from csv file: ", data_dir_csv)
                with open(data_dir_csv, 'r') as f:
                    reader = csv.reader(f)
                    image_paths = []
                    for row in reader:
                        img_path = os.path.join(data_dir, row[0])
                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                    self.paths.extend(image_paths)
                    print("number of images: ", len(image_paths))
            else:
                rgb_fnames = os.listdir(osp.join(data_dir, "images_rgb"))
                rgb_fnames = [f'{data_dir}/images_rgb/{e}' for e in rgb_fnames]
                self.paths.extend(rgb_fnames)
            
            # process synonyms if there exists synonyms 
            synonyms_fp = os.path.join(data_dir, "synonyms.json")
            if os.path.exists(synonyms_fp) and self.replace_synonyms:
                print("loading synonyms from: ", synonyms_fp)
                with open(synonyms_fp, "r") as f:
                    synonyms = json.load(f)
                    for k in synonyms:
                        if k in self.synonyms:
                            self.synonyms[k] = list(set(self.synonyms[k] + synonyms[k]))
                        else:
                            self.synonyms[k] = list(set([k] + synonyms[k])) # Just fixed this, no model trained with this + keep k syn 

        if keep_k_synonyms is not None:
            print("keeping only the top k synonyms: ", keep_k_synonyms)
            for k in self.synonyms:
                self.synonyms[k] = self.synonyms[k][:keep_k_synonyms]

        print("number of unique words in synonyms: ", len(self.synonyms))
        print("total number of unique words in synonmys: ", len(set(chain.from_iterable(self.synonyms.values()))))
        
        # Split dataset
        if split in ["train", "val"]:
            train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)
        if split == 'train':
            print("number of training samples: ", len(train_paths))
        elif split == 'val':
            print("number of testing samples: ", len(test_paths))
        elif split == "test":
            print("number of testing samples: ", len(self.paths))
        if split == 'train':
            self.paths = train_paths
        elif split == 'val':
            self.paths = test_paths
        elif split == "test":
            # just to show that self.paths are unchanged
            self.paths = self.paths
    
    def get_tactile_path(self, rgb_path):
        return rgb_path.replace("rgb", "tac")

    def get_text_path(self, rgb_path):
        id = re.search(r'image_(\d+)_rgb', rgb_path).group(1)
        data_dir = "/".join(rgb_path.split("/")[:-2])
        return f"{data_dir}/text/labels_{id}.txt"

    def __len__(self):
        return len(self.paths)
    
    def load_vision_data(self, path : str):
        return load_vision_data(path, self.rgb_size, self.im_scale_range, transform_rgb=self.transform_rgb, dataset_version="v1")

    def load_tactile_data(self, path: str):
        return load_tactile_data(path, transform_tac=self.transform_tac)
    
    def load_text_data(self, path: str):
        if self.replace_synonyms:
            return load_text_data(path, shuffle=self.shuffle_text, prompt=self.text_prompt, synonyms_dict=self.synonyms)
        return load_text_data(path, shuffle=self.shuffle_text, prompt=self.text_prompt)

    def __getitem__(self, index):
        img_path = self.paths[index]
        item = OrderedDict()
        if ModalityType.VISION in self.modality_types:
            images = self.load_vision_data(img_path)
            item[ModalityType.VISION] = [images]
        if ModalityType.TACTILE in self.modality_types:
            tactiles = self.load_tactile_data(self.get_tactile_path(img_path))
            item[ModalityType.TACTILE] = [tactiles]
        if ModalityType.TEXT in self.modality_types:
            texts = self.load_text_data(self.get_text_path(img_path))
            item[ModalityType.TEXT] = texts
        return item
