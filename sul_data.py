"""Image-text datasets, image transforms, BERT tokenization, and batching."""

import json
import os
import random
from pathlib import Path

import torch
import torch.utils.data as data
from torchvision.transforms import RandomResizedCrop, InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Counts required by the submitted paper's COCO protocol.
COCO_SPLIT_COUNTS = {
    'train': (113287, 566435),
    'dev': (5000, 25000),
    'testall': (5000, 25000),
}


def validate_coco_split(split, image_ids, captions):
    """Validate the provided split; this function does not construct new splits."""
    if split not in COCO_SPLIT_COUNTS:
        raise ValueError(f'Unknown COCO split: {split}')
    expected_images, expected_captions = COCO_SPLIT_COUNTS[split]
    if len(set(image_ids)) != expected_images or len(captions) != expected_captions:
        raise ValueError(
            f'COCO {split} requires {expected_images} distinct images and '
            f'{expected_captions} captions; got {len(set(image_ids))} images and '
            f'{len(captions)} captions. Supply the corresponding annotation files.')
    if split == 'train':
        if len(image_ids) != expected_images or len(image_ids) != len(set(image_ids)):
            raise ValueError('COCO train requires one unique image ID per five captions')
    elif len(image_ids) != len(captions) or any(
            len(set(image_ids[i:i + 5])) != 1 for i in range(0, len(image_ids), 5)):
        raise ValueError('COCO evaluation requires five consecutive caption rows per image')


class ImageTransform:
    def __init__(self, size=224, train=True):
        self.size = size
        self.train = train

    def __call__(self, image):
        if self.train:
            top, left, height, width = RandomResizedCrop.get_params(
                image, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3))
            flip = bool(torch.rand(1) < 0.5)
        else:
            top, left, height, width = 0, 0, image.height, image.width
            flip = False
        image = TF.resized_crop(
            image, top, left, height, width, [self.size, self.size],
            interpolation=InterpolationMode.BICUBIC)
        if flip:
            image = TF.hflip(image)
        return TF.normalize(TF.to_tensor(image),
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def process_caption_bert(tokenizer, tokens, train=True, mask_rate=0.2, size_augment=True):
    """Retain the active text augmentation and all non-PAD BERT positions."""
    output_tokens = []
    deleted_idx = []
    for token in tokens:
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()
        if size_augment and prob < mask_rate and train:
            prob /= mask_rate
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append('[MASK]')
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                output_tokens.append(sub_token)
    if deleted_idx:
        output_tokens = [token for i, token in enumerate(output_tokens) if i not in deleted_idx]
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    return torch.tensor(tokenizer.convert_tokens_to_ids(output_tokens))


class RawImageDataset(data.Dataset):
    def __init__(self, opt, data_path, split, tokenizer, train):
        self.opt = opt
        self.train = train
        self.split = split
        self.tokenizer = tokenizer
        annotation_root = Path(opt.coco_annotations_path) if opt.dataset == 'coco' else Path(data_path) / opt.dataset
        self.image_base = opt.f30k_img_path if opt.dataset == 'f30k' else opt.coco_img_path
        mapping_name = 'id_mapping_2017.json' if opt.dataset == 'coco' else 'id_mapping.json'
        with (annotation_root / mapping_name).open(encoding='utf-8') as stream:
            self.id_to_path = json.load(stream)
        self.captions = [line.strip() for line in
                         (annotation_root / f'{split}_caps.txt').read_text(encoding='utf-8').splitlines()]
        self.images = [int(line.strip()) for line in
                       (annotation_root / f'{split}_ids.txt').read_text(encoding='utf-8').splitlines()]
        self.length = len(self.captions)
        self.im_div = 5 if len(self.images) != self.length else 1
        if opt.dataset == 'coco':
            validate_coco_split(split, self.images, self.captions)
        self.preprocess = ImageTransform(size=opt.img_res, train=train)

    def __getitem__(self, index):
        img_index = index // self.im_div
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(self.captions[index])
        caption = process_caption_bert(
            self.tokenizer, basic_tokens, self.train, size_augment=self.opt.size_augment)
        image_id = self.images[img_index]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        with Image.open(image_path) as raw:
            image = raw.convert('RGB')
        # Keep the caption row index separate from the physical image identity.
        return self.preprocess(image), caption, index, image_id

    def __len__(self):
        return self.length


def collate_fn_ours(rows):
    rows.sort(key=lambda row: len(row[1]), reverse=True)
    images, captions, row_ids, image_ids = zip(*rows)
    lengths = torch.tensor([len(caption) for caption in captions])
    padded = torch.zeros(len(captions), int(lengths.max())).long()
    for i, caption in enumerate(captions):
        padded[i, :len(caption)] = caption
    return (torch.stack(images), padded, lengths,
            torch.tensor(row_ids), torch.tensor(image_ids))


def get_loader(opt, data_path, split, tokenizer, batch_size=128,
               shuffle=True, num_workers=2, train=True):
    dataset = RawImageDataset(opt, data_path, split, tokenizer, train)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn_ours, drop_last=train,
        persistent_workers=bool(opt.persistent_workers and num_workers > 0))


def get_train_loader(opt, data_path, tokenizer, batch_size, workers, split='train'):
    return get_loader(opt, data_path, split, tokenizer, batch_size, True, workers, train=True)


def get_test_loader(opt, data_path, tokenizer, batch_size, workers, split='test'):
    if opt.dataset == 'coco' and split == 'test':
        split = 'testall'
    return get_loader(opt, data_path, split, tokenizer, batch_size, False, workers, train=False)
