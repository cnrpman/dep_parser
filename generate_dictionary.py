import pickle
import json

with open("train.converted", 'rb') as f:
    configs = pickle.load(f)
lemmas = []
postags = []
arclabels = []
arcrels = []
for config in configs:
    sw, st, sl, arc, relation = config
    lemmas.extend(sw)
    postags.extend(st)
    arcrels.extend(sl)
    arclabels.append(arc)
    arcrels.append(relation)

lemmas = list(set(lemmas))
postags = list(set(postags))
arclabels = list(set(arclabels))
arcrels = list(set(arcrels))

with open("dictionary.json", 'w') as f:
    json.dump([lemmas, postags, arclabels, arcrels], f)
