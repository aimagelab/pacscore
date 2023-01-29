import argparse
import torch
import evaluation
import scipy.stats

from models.clip import clip
from utils import collate_fn
from evaluation import PACScore, RefPACScore
from models import open_clip
from data import Flickr8k
from torch.utils.data import DataLoader


_MODELS = {
    "ViT-B/32": "checkpoints/clip_ViT-B-32.pth",
    "open_clip_ViT-L/14": "checkpoints/openClip_ViT-L-14.pth"
}

def compute_correlation_scores(dataloader, model, preprocess, args):
    gen = {}
    gts = {}

    human_scores = list()
    ims_cs = list()
    gen_cs = list()
    gts_cs = list()
    all_scores = dict()
    model.eval()

    for it, (images, candidates, references, scores) in enumerate(iter(dataloader)):
        for i, (im_i, gts_i, gen_i, score_i) in enumerate(zip(images, references, candidates, scores)):
            gen['%d_%d' % (it, i)] = [gen_i, ]
            gts['%d_%d' % (it, i)] = gts_i

            ims_cs.append(im_i)
            gen_cs.append(gen_i)
            gts_cs.append(gts_i)
            human_scores.append(score_i)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    all_scores_metrics = evaluation.get_all_metrics(gts, gen, return_per_cap=True)
    
    for k, v in all_scores_metrics.items():
        if k == 'BLEU':
            all_scores['BLEU-1'] = v[0]
            all_scores['BLEU-4'] = v[-1]
        else:
            all_scores[k] = v
    
    # PAC-S
    _, pac_scores, candidate_feats, len_candidates = PACScore(model, preprocess, ims_cs, gen_cs, device, w=2.0)
    all_scores['PAC-S'] = pac_scores
    
    # RefPAC-S
    if args.compute_refpac:
        _, per_instance_text_text = RefPACScore(model, gts_cs, candidate_feats, device, torch.tensor(len_candidates))
        refpac_scores = 2 * pac_scores * per_instance_text_text / (pac_scores + per_instance_text_text)
        all_scores['RefPAC-S'] = refpac_scores

    for k, v in all_scores.items():
        kendalltau_b = 100 * scipy.stats.kendalltau(v, human_scores, variant='b')[0]
        kendalltau_c = 100 * scipy.stats.kendalltau(v, human_scores, variant='c')[0]
        print('%s \t Kendall Tau-b: %.3f \t  Kendall Tau-c: %.3f'
              % (k, kendalltau_b, kendalltau_c))


def compute_scores(model, preprocess, args):
    args.datasets = ['flickr8k_expert', 'flickr8k_cf'] 

    args.batch_size_compute_score = 10
    for d in args.datasets:
        print("Computing correlation scores on dataset: " + d)
        if d == 'flickr8k_expert':
            dataset = Flickr8k(json_file='flickr8k.json')
            dataloader = DataLoader(dataset, batch_size=args.batch_size_compute_score, shuffle=False, collate_fn=collate_fn)
        elif d == 'flickr8k_cf':
            dataset = Flickr8k(json_file='crowdflower_flickr8k.json')
            dataloader = DataLoader(dataset, batch_size=args.batch_size_compute_score, shuffle=False, collate_fn=collate_fn)

        compute_correlation_scores(dataloader, model, preprocess, args)



if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', 
                    choices=['ViT-B/32', 'open_clip_ViT-L/14'])    
    parser.add_argument('--compute_refpac', action='store_true')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.clip_model.startswith('open_clip'):
        print("Using Open CLIP Model: " + args.clip_model)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    else:
        print("Using CLIP Model: " + args.clip_model)
        model, preprocess = clip.load(args.clip_model, device=device) 
        
    model = model.to(device)
    model = model.float()
    
    checkpoint = torch.load(_MODELS[args.clip_model])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    compute_scores(model, preprocess, args)
    