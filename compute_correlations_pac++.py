import torch
import argparse
from models.clip_lora import clip_lora
from compute_correlations import compute_scores

_MODELS = {
    "ViT-B/32": "checkpoints/PAC++_clip_ViT-B-32.pth",
    "ViT-L/14": "checkpoints/PAC++_clip_ViT-L-14.pth"
}

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='PAC-S evaluation')
    parser.add_argument('--compute_refpac', action='store_true')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-L/14'])

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_r = 4
    args.clip_model = 'ViT-L/14'
    print("Using CLIP Model: " + args.clip_model)
    model, preprocess = clip_lora.load(
        args.clip_model, device=device, lora=lora_r)

    model = model.to(device)
    model = model.float()

    checkpoint = torch.load(_MODELS[args.clip_model])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    compute_scores(model, preprocess, args, device)
