from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import collections
from models import clip


class CapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if transform:
            self.preprocess = transform
        else:
            self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=6):
    data = torch.utils.data.DataLoader(CapDataset(captions), batch_size=batch_size, num_workers=num_workers,
                                       shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).detach().cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, transform, device, batch_size=64, num_workers=6):
    data = torch.utils.data.DataLoader(ImageDataset(images, transform), batch_size=batch_size, num_workers=num_workers,
                                       shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            all_image_features.append(model.encode_image(b).detach().cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def PACScore(model, transform, images, candidates, device, w=2.0):
    '''
    compute the unreferenced PAC score.
    '''
    len_candidates = [len(c.split()) for c in candidates] 
    if isinstance(images, list):
        # extracting image features
        images = extract_all_images(images, model, transform, device)

    candidates = extract_all_captions(candidates, model, device)

    images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
    candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates, len_candidates


def RefPACScore(model, references, candidates, device, len_candidates):
    '''
    compute the RefPAC score, extracting only the reference captions.
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    len_references = []
    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        len_r = [len(r.split()) for r in refs]
        len_references.append(len_r)
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)
    
    candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))
    flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs ** 2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, (cand, l_ref, l_cand) in enumerate(zip(candidates, len_references, len_candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())

        per.append(np.max(all_sims))

    return np.mean(per), per


