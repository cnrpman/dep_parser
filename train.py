import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import ConfigDataset
from model import FPN
from utils import hmean

def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', default="train.converted", help='path to converted train configuration')
    parser.add_argument('--path_dev', default="dev.converted", help='path to converted dev configuration')
    parser.add_argument('--path_test', default="test.converted", help='path to converted test configuration')
    parser.add_argument('--path_dictionary', default='dictionary.json', help='path to dictionary file')
    parser.add_argument('--batchsize', default=10000, type=int, help='batch size to use')
    parser.add_argument('--iters', default=20000, type=int, help='num of iters to use')

    parser.add_argument('--embedding_dim', default=50, type=int, help='dimension of embedding to use')
    parser.add_argument('--feature_size', default=48, type=int, help='size of feature to use (elements in sw, st and sl)')
    parser.add_argument('--hidden_dim', default=200, type=int, help='dimension of hidden layer to use')
    parser.add_argument('--cubic', action="store_true", help='use cube activation instead of tanh')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for ada grad')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weight decay rate (regularization)')
    parser.add_argument('--drop_out', default=0.5, type=float, help='dropout rate to use')
    parser.add_argument('--model_path', default="train.model", help="path for saving model")
    parser.add_argument('--init_range', default=-1, type=float, help='range for weight init (uniform)')
    parser.add_argument('--splitlabel', action="store_true", help='predict transit and relation independently')
    parser.add_argument('--pretrained_embedding', default="", help='provide the path to the pretrained embedding if you want (should have the same dim as --embedding_dim), or use learnable embedding by default')
    parser.add_argument('--pretrained_embedding_unk', default="", help='provide the path to the pretrained unk embedding if you want (should have the same dim as --embedding_dim), or use learnable embedding by default')

    return parser.parse_args()

def main():
    args = opts()
    trainset = ConfigDataset(args.path_train, args.path_dictionary, args.pretrained_embedding, args.pretrained_embedding_unk)
    devset = ConfigDataset(args.path_dev, args.path_dictionary, args.pretrained_embedding, args.pretrained_embedding_unk)

    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=8)
    devloader = DataLoader(devset, batch_size=args.batchsize, num_workers=8)

    loss_function = nn.CrossEntropyLoss()
    model = FPN(
        args,
        vocab_size=len(trainset.lemmas_to_ix), 
        postag_size=len(trainset.postags_to_ix),
        rel_size=len(trainset.arcrels_to_ix),
        transit_size=len(trainset.arclabels_to_ix) if args.splitlabel else len(trainset.combine_to_ix)
    )

    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # log down metrics at each iteration
    train_metrics = {
        "loss": [],
        "acc_transit": [],
        "acc_relation": []
    }
    # log down metrics for whole epoch
    dev_metrics = {
        "acc_transit": [],
        "acc_relation": []
    }

    def train(model, trainloader, metrics, remains_iter):
        for b in tqdm(trainloader):
            model.train()
            model.zero_grad()
            transit_crit = b['transit'] if args.splitlabel else b['combine']

            logits_tran, logits_rel = model(b['sw'].cuda(), b['st'].cuda(), b['sl'].cuda())

            loss_tran = loss_function(logits_tran, transit_crit.cuda())
            loss_rel = loss_function(logits_rel, b['relation'].cuda())
            losses = (loss_tran + loss_rel) if args.splitlabel else loss_tran
            losses.backward()
            optimizer.step()

            metrics["loss"].append(torch.mean(losses, dim=0).item())
            metrics["acc_transit"].append(torch.mean((torch.argmax(logits_tran, dim=1) == transit_crit.cuda()).float(), dim=0).item())
            metrics["acc_relation"].append(torch.mean((torch.argmax(logits_rel, dim=1) == b['relation'].cuda()).float(), dim=0).item())

            remains_iter -= 1
            if remains_iter < 0:
                break

    def evalu(model, evalloader, metrics):
        acc_transit = []
        acc_relation = []
        for b in evalloader:
            model.eval()
            transit_crit = b['transit'] if args.splitlabel else b['combine']

            logits_tran, logits_rel = model(b['sw'].cuda(), b['st'].cuda(), b['sl'].cuda())
            acc_transit.append(torch.sum((torch.argmax(logits_tran, dim=1) == transit_crit.cuda()).float(), dim=0).item())
            acc_relation.append(torch.sum((torch.argmax(logits_rel, dim=1) == b['relation'].cuda()).float(), dim=0).item())
        metrics['acc_transit'].append(np.sum(acc_transit) / len(evalloader.dataset))
        metrics['acc_relation'].append(np.sum(acc_relation) / len(evalloader.dataset))

    best_acc = -1
    cur_iter = 0
    idx = 0
    while cur_iter < args.iters:
        train(model, trainloader, train_metrics, args.iters-cur_iter)
        print("Epoch %d Iter %d [Train set] Loss: %.5f \t Accuracy Transit: %.2f%% \t Accuracy Relation: %.2f%%" % (idx, cur_iter, np.average(train_metrics["loss"][-5:]), np.average(train_metrics["acc_transit"][-5:]) * 100,  np.average(train_metrics["acc_relation"][-5:]) * 100))

        evalu(model, devloader, dev_metrics)
        print("Epoch %d Iter %d [Dev set] Accuracy Transit: %.2f%% \t Accuracy Relation: %.2f%%" % (idx, cur_iter, dev_metrics["acc_transit"][-1] * 100,  dev_metrics["acc_relation"][-1] * 100))

        acc_mean = hmean(dev_metrics["acc_transit"][-1], dev_metrics["acc_relation"][-1]) if args.splitlabel else dev_metrics["acc_transit"][-1]
        if acc_mean > best_acc:
            print("Saving best model! Now: %.2f%% <- Previous: %.2f%%" % (acc_mean*100, best_acc*100))
            torch.save({
                "args": args,
                "model_state_dict": model.state_dict(),
            }, args.model_path)
            best_acc = acc_mean
        
        cur_iter += len(trainloader)
        idx += 1

if __name__ == "__main__":
    main()