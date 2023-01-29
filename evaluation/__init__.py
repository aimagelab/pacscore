'''
Automatic generation evaluation metrics wrapper
The most useful function here is
get_all_metrics(refs, cands)
'''
from .pac_score import PACScore, RefPACScore
from .tokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

def get_all_metrics(refs, cands, return_per_cap=False):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(4), 'BLEU'),
                               (Meteor(), 'METEOR'),
                               (Rouge(), 'ROUGE'),
                               (Cider(), 'CIDER'),
                            #    (Spice(), 'SPICE')
                               ]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        if return_per_cap:
            metrics.append(per_cap)
        else:
            metrics.append(overall)
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics


def pycoco_eval(scorer, refs, cands):
    '''
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    '''
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores
