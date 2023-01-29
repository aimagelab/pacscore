import torch
import os
from PIL import Image
import json
import numpy as np
import torch

class Flickr8k(torch.utils.data.Dataset):
    def __init__(self, json_file, root='datasets/flickr8k/',
                 transform=None, load_images=False):
        self.im_folder = os.path.join(root, 'images')
        self.transform = transform
        self.load_images = load_images

        with open(os.path.join(root, json_file)) as fp:
            data = json.load(fp)

        self.data = list()
        for i in data:
            for human_judgement in data[i]['human_judgement']:
                if np.isnan(human_judgement['rating']):
                    print('NaN')
                    continue
                d = {
                    'image': data[i]['image_path'].split('/')[-1],
                    'references': [' '.join(gt.split()) for gt in data[i]['ground_truth']],
                    'candidate': ' '.join(human_judgement['caption'].split()),
                    'human_score': human_judgement['rating']
                }
                self.data.append(d)

    def get_image(self, filename):
        img = Image.open(os.path.join(self.im_folder, filename)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_idx = self.data[idx]['image']
        candidate = self.data[idx]['candidate']
        references = self.data[idx]['references']
        score = self.data[idx]['human_score']

        if self.load_images:
            im = self.get_image(im_idx)
        else:
            im = os.path.join(self.im_folder, im_idx)

        return im, candidate, references, score



