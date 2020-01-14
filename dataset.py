import pickle
import json
from itertools import product

import torch
from torch.utils.data import Dataset

UNK_token = "UNK"

class ConfigDataset(Dataset):
    def __init__(self, path, dictionary, word_embedding="", word_embedding_unk=""):
        self.path = path
        if self.path is not None:
            with open(path, 'rb') as f:
                self.configs = pickle.load(f)
        with open(dictionary, 'r') as f2:
            lemmas, postags, arclabels, arcrels = json.load(f2)

        if word_embedding != "" and word_embedding_unk != "":
            self.lemmas_to_ix = {}
            self.ix_to_lemmas = []
            with open(word_embedding, "r") as f:
                for line in f:
                    row = line.strip().split()
                    self.lemmas_to_ix[row[0]] = len(self.ix_to_lemmas)
                    self.ix_to_lemmas.append(row[0])
            self.lemmas_to_ix["UNK"] = len(self.ix_to_lemmas)
            self.ix_to_lemmas += ["UNK"]
        else:
            self.lemmas_to_ix = {lemma: i for i, lemma in enumerate(lemmas)}
            self.lemmas_to_ix["UNK"] = len(self.lemmas_to_ix)
            self.ix_to_lemmas = lemmas + ["UNK"]
        self.postags_to_ix = {postag: i for i, postag in enumerate(postags)}
        self.ix_to_postags = postags
        self.arclabels_to_ix = {arclabel: i for i, arclabel in enumerate(arclabels)}
        self.ix_to_arclabels = arclabels
        self.arcrels_to_ix = {arcrel: i for i, arcrel in enumerate(arcrels)}
        self.ix_to_arcrels = arcrels

        self.combine_to_ix = {}
        self.ix_to_combine = []
        for arclabel, arcrel in product(arclabels, arcrels):
            symbol = arclabel + '|' + arcrel
            self.combine_to_ix[symbol] = len(self.ix_to_combine)
            self.ix_to_combine.append(symbol)
    
    def __len__(self):
        return len(self.configs)

    def convert(self, config):
        sw, st, sl, arc, relation = config
        return {
            "sw": torch.tensor([self.lemmas_to_ix[w] if w in self.lemmas_to_ix else self.lemmas_to_ix[UNK_token] for w in sw], dtype=torch.long),
            "st": torch.tensor([self.postags_to_ix[t] for t in st], dtype=torch.long),
            "sl": torch.tensor([self.arcrels_to_ix[r] for r in sl], dtype=torch.long),
            "transit": torch.tensor(self.arclabels_to_ix[arc], dtype=torch.long),
            "relation": torch.tensor(self.arcrels_to_ix[relation], dtype=torch.long),
            "combine": torch.tensor(self.combine_to_ix[arc + '|' + relation], dtype=torch.long)
        }

    def __getitem__(self, idx):
        return self.convert(self.configs[idx])

if __name__ == "__main__":
    dataset = ConfigDataset("train.converted", "dictionary.json")
    print(dataset[0]["sw"], dataset[0]["relation"])