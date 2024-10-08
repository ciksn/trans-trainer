import torch
import numpy
from torchvision import transforms
from typing import List, Any
from PIL import Image
import random

from .randaugment import RandomAugment

class ImageCaptionProcessor:
    def __init__(self, image_size=224, min_scale = 0.5, randaug=False, task_type='pretrain'):
        self.image_size = image_size
        self.min_scale = min_scale
        self.task_type= task_type

        if randaug:
            self.image_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.image_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.text_transform = None

    def __call__(self, image, text) -> tuple[torch.Tensor, Any]:
        assert image or text
        
        if image:
            image_input = self.image_transform(image)
        else:
            image_input = None

        if text and not isinstance(text, str):
            if isinstance(text["prompt"], list):
                prompt = random.choice(text["prompt"])
            else:
                prompt = text["prompt"]
            text_input = dict(
                prompt=prompt,
                completion=text["text"],
            )
        elif isinstance(text, str):
            text_input = text
        else:
            text_input = None
        return image_input, text_input
    
class ImageCaptionProcessorWithoutCrop:
    def __init__(self, image_size=224, min_scale = 0.5, randaug=False, task_type='pretrain'):
        self.image_size = image_size
        self.min_scale = min_scale
        self.task_type= task_type

        if randaug:
            self.image_transform = transforms.Compose([
                # transforms.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                transforms.Resize([image_size,image_size],interpolation=Image.BICUBIC),
                # transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.image_transform = transforms.Compose([
                # transforms.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                transforms.Resize([image_size,image_size],interpolation=Image.BICUBIC),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.text_transform = None

    def __call__(self, image, text) -> tuple[torch.Tensor, Any]:
        if image is not None:
            image_input = self.image_transform(image)
        else:
            image_input = None

        if text is not None and not isinstance(text, str):
            if isinstance(text["prompt"], list):
                prompt = random.choice(text["prompt"])
            else:
                prompt = text["prompt"]
            text_input = dict(
                prompt=prompt,
                completion=text["text"],
            )
        elif isinstance(text, str):
            text_input = text
        else:
            text_input = None
        return image_input, text_input