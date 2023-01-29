import os
import argparse
import torch
import json
import evaluation
import numpy as np

from models.clip import clip
from evaluation import PACScore, RefPACScore
from models import open_clip

_MODELS = {
    "ViT-B/32": "checkpoints/clip_ViT-B-32.pth",
    "open_clip_ViT-L/14": "checkpoints/openClip_ViT-L-14.pth"
}


def compute_scores(model, preprocess, image_ids, candidates, references, args):
    gen = {}
    gts = {}

    ims_cs = list()
    gen_cs = list()
    gts_cs = list()
    all_scores = dict()
    model.eval()

    for i, (im_i, gts_i, gen_i) in enumerate(zip(image_ids, references, candidates)):
        gen['%d' % (i)] = [gen_i, ]
        gts['%d' % (i)] = gts_i

        ims_cs.append(im_i)
        gen_cs.append(gen_i)
        gts_cs.append(gts_i)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    all_scores_metrics = evaluation.get_all_metrics(gts, gen)
    
    for k, v in all_scores_metrics.items():
        if k == 'BLEU':
            all_scores['BLEU-1'] = v[0]
            all_scores['BLEU-4'] = v[-1]
        else:
            all_scores[k] = v

    # PAC-S
    _, pac_scores, candidate_feats, len_candidates = PACScore(
        model, preprocess, ims_cs, gen_cs, device, w=2.0)
    all_scores['PAC-S'] = np.mean(pac_scores)

    # RefPAC-S
    if args.compute_refpac:
        _, per_instance_text_text = RefPACScore(
            model, gts_cs, candidate_feats, device, torch.tensor(len_candidates))
        refpac_scores = 2 * pac_scores * per_instance_text_text / \
            (pac_scores + per_instance_text_text)
        all_scores['RefPAC-S'] = np.mean(refpac_scores)

    return all_scores


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'open_clip_ViT-L/14'])
    parser.add_argument('--image_dir', type=str, default='example/images')
    parser.add_argument('--candidates_json', type=str,
                        default='example/good_captions.json')
    parser.add_argument('--references_json', type=str, default='example/refs.json')
    parser.add_argument('--compute_refpac', action='store_true')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_ids = [img_id for img_id in os.listdir(args.image_dir)]

    with open(args.candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid.split('.')[0]] for cid in image_ids]

    with open(args.references_json) as f:
        references = json.load(f)
        references = [references[cid.split('.')[0]] for cid in image_ids]

    image_ids = [os.path.join(args.image_dir, img_id) for img_id in image_ids]

    if args.clip_model.startswith('open_clip'):
        print("Using Open CLIP Model: " + args.clip_model)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k')
    else:
        print("Using CLIP Model: " + args.clip_model)
        model, preprocess = clip.load(args.clip_model, device=device)

    model = model.to(device)
    model = model.float()

    checkpoint = torch.load(_MODELS[args.clip_model])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_scores = compute_scores(
        model, preprocess, image_ids, candidates, references, args)
    
    for k, v in all_scores.items():
        print('%s: %.4f' % (k, v))
