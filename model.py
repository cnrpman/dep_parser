import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FPN(nn.Module):

    def __init__(self, args, vocab_size, postag_size, rel_size, transit_size):
        super().__init__()
        self.args = args
        
        if args.pretrained_embedding != "" and args.pretrained_embedding_unk != "":
            embedding_array = []
            with open(args.pretrained_embedding, "r") as f:
                for line in f:
                    row = line.strip().split()
                    embedding_array.append(np.array([float(r) for r in row[1:]], dtype=np.float32))

            with open(args.pretrained_embedding_unk, "r") as f:
                for line in f:
                    row = line.strip().split()
                embedding_array.append(np.array([float(r) for r in row[1:]], dtype=np.float32))
            pretrained_weight = torch.Tensor(embedding_array)
            self.embedding_w = nn.Embedding.from_pretrained(pretrained_weight)
        else:
            self.embedding_w = nn.Embedding(vocab_size, args.embedding_dim)
            if args.init_range != -1:
                self.embedding_w.weight.data.uniform_(-args.init_range, args.init_range)

        self.embedding_t = nn.Embedding(postag_size, args.embedding_dim)
        if args.init_range != -1:
            self.embedding_t.weight.data.uniform_(-args.init_range, args.init_range)
        self.embedding_l = nn.Embedding(rel_size, args.embedding_dim)
        if args.init_range != -1:
            self.embedding_l.weight.data.uniform_(-args.init_range, args.init_range)

        self.dropout_emb = nn.Dropout(p=args.drop_out)
        self.linear1 = nn.Linear(args.feature_size * args.embedding_dim, args.hidden_dim)
        self.dropout_hid = nn.Dropout(p=args.drop_out)
        self.linear_tran = nn.Linear(args.hidden_dim, transit_size)
        self.linear_rel = nn.Linear(args.hidden_dim, rel_size)

    def forward(self, sw, st, sl):
        embeds_w = self.embedding_w(sw).view((sw.size(0), -1))
        embeds_t = self.embedding_t(st).view((st.size(0), -1))
        embeds_l = self.embedding_l(sl).view((sl.size(0), -1))
        embeds = self.dropout_emb(torch.cat((embeds_w, embeds_t, embeds_l), 1))

        preact = self.linear1(embeds)
        if hasattr(self.args, 'cubic') and self.args.cubic:
            act = torch.pow(preact, 3)
        else:
            act = torch.tanh(preact)
        hid = self.dropout_hid(act)

        logits_tran = self.linear_tran(hid)
        logits_rel = self.linear_rel(hid)
        return logits_tran, logits_rel