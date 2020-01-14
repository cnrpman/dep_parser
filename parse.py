import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from preparedata import read_conll, generate_step, ROOT_symbol, ARC_LEFT_symbol, ARC_RIGHT_symbol
from dataset import ConfigDataset
from model import FPN
from utils import hmean

def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputfile', '-o', default="test.out", help='path to converted dev configuration')
    parser.add_argument('--inputfile', '-i', default="test.orig.conll", help='path to converted test configuration')
    parser.add_argument('--modelfile', '-m', default="train.model", help="path for saving model")

    return parser.parse_args()

def main():
    args = opts()
    checkpoint = torch.load(args.modelfile)
    c_args = checkpoint['args']
    dummyset = ConfigDataset(None, c_args.path_dictionary, c_args.pretrained_embedding, c_args.pretrained_embedding_unk)

    model = FPN(
        c_args,
        vocab_size=len(dummyset.lemmas_to_ix), 
        postag_size=len(dummyset.postags_to_ix),
        rel_size=len(dummyset.arcrels_to_ix),
        transit_size=len(dummyset.arclabels_to_ix) if c_args.splitlabel else len(dummyset.combine_to_ix)
    )

    model = nn.DataParallel(model)
    model = model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(args.inputfile)
    samples = read_conll(args.inputfile)

    f_out = open(args.outputfile, "w")

    for sample in tqdm(samples):
        stack = [{
            "index": 0,
            "token": ROOT_symbol,
            "lemma": ROOT_symbol,
            "pos_tag": ROOT_symbol,
            "parent": 0,
            "relation": ROOT_symbol,
            "left_children": [],
            "right_children": []
        }]

        buffer = copy.copy(sample)

        while True:
            feats = generate_step([stack, buffer, dummyset.ix_to_arclabels[0], dummyset.ix_to_arcrels[0]])
            torch_sample = dummyset.convert(feats)
            logits_tran, logits_rel = model(torch.unsqueeze(torch_sample['sw'], 0).cuda(), torch.unsqueeze(torch_sample['st'], 0).cuda(), torch.unsqueeze(torch_sample['sl'], 0).cuda())
            if not c_args.splitlabel:
                transit, relation = dummyset.ix_to_combine[torch.argmax(logits_tran, dim=1)[0].item()].split('|')
            else:
                transit = dummyset.ix_to_arclabels[torch.argmax(logits_tran, dim=1)[0].item()]
                relation = dummyset.ix_to_arcrels[torch.argmax(logits_rel, dim=1)[0].item()]
            
            if transit == ARC_LEFT_symbol and len(stack) > 1:
                stack[-1]["left_children"].append(stack[-2])
                stack[-2]['parent'] = stack[-1]["index"]
                stack[-2]['relation'] = relation
                stack = stack[:-2] + [stack[-1]]
            elif transit == ARC_RIGHT_symbol and len(stack) > 1:
                stack[-2]["right_children"].append(stack[-1])
                stack[-1]['parent'] = stack[-2]["index"]
                stack[-1]['relation'] = relation
                stack = stack[:-1]
            elif len(buffer) > 0:
                stack.append(buffer[0])
                buffer = buffer[1:]
            elif len(stack) > 1:
                stack[-2]["right_children"].append(stack[-1])
                stack[-1]['parent'] = stack[-2]["index"]
                stack[-1]['relation'] = relation
                stack = stack[:-1]
            else:
                break

        for token in sample:
            f_out.write("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n" % (token["index"], token["token"], token["lemma"], token["pos_tag"], token["pos_tag"], "_", token["parent"], token["relation"], "_", "_"))
        f_out.write("\n")

if __name__ == "__main__":
    main()