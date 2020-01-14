import sys
from collections import Counter, defaultdict
import copy
from tqdm import tqdm
import pickle

ROOT_symbol = "[ROOT]"
ARC_RIGHT_symbol = "ARC_RIGHT"
ARC_LEFT_symbol = "ARC_LEFT"
SHIFT_symbol = "SHIFT"
NOREL_symbol = "NOREL"
NULL_symbol = "[NULL]"
POS_prefix = "[POS]"
LABEL_prefix = "[LABEL]"

TOP_N = 3
TOP_N_C = 2
TOP_N_C_N = 2

def read_conll(path):
    """
    Read list of samples
    """
    samples = []
    with open(path, 'r') as f:
        sen = []
        for line in f:
            line = line.strip()
            if len(line):
                idx, tok, lem, pos, _, _, par, rel, _, _ = line.split('\t')
                sen.append({
                    "index": int(idx),
                    "token": tok,
                    "lemma": lem,
                    "pos_tag": pos,
                    "parent": int(par),
                    "relation": rel,
                    "left_children": [],
                    "right_children": []
                })
            else:
                if len(sen):
                    samples.append(sen)
                sen = []
        if len(sen):
            samples.append(sen)

    return samples

def samples_stat(samples):
    """
    Print stats of sample
    """
    print("Number of sample:", len(samples))

    to_collect = ['lemma', 'pos_tag', 'relation']
    cols = {name: Counter() for name in to_collect}
    for sample in samples:
        for token in sample:
            for name in to_collect:
                val = token[name]
                cols[name][val] += 1
    for name in to_collect:
        print("stat", name, ":")
        if name is not 'lemma':
            print(len(cols[name]))
            print(cols[name])
        else:
            print(len(cols[name]))
            print_distribution(cols[name])

def print_distribution(the_dict):
    bins = Counter()
    for lemma in the_dict:
        num = the_dict[lemma]
        bins[num] += 1
    print(bins)

nonprojective = 0

def create_config(sample):
    """
    given a sample, create configs
    """
    global nonprojective

    remain_childs = [0] + [0 for token in sample] # Log number of children
    for token in sample:
        remain_childs[token["parent"]] += 1

    configs = []
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

    def shift():
        nonlocal buffer
        configs.append([copy.deepcopy(stack), copy.copy(buffer), SHIFT_symbol, NOREL_symbol])
        stack.append(buffer[0])
        buffer = buffer[1:]
    def arc_right():
        nonlocal remain_childs, configs, stack
        configs.append([copy.deepcopy(stack), copy.copy(buffer), ARC_RIGHT_symbol, stack[-1]["relation"]])
        remain_childs[stack[-2]["index"]] -= 1
        stack[-2]["right_children"].append(stack[-1])
        stack = stack[:-1]
    def arc_left():
        nonlocal remain_childs, configs, stack
        configs.append([copy.deepcopy(stack), copy.copy(buffer), ARC_LEFT_symbol, stack[-2]["relation"]])
        remain_childs[stack[-1]["index"]] -= 1
        stack[-1]["left_children"].append(stack[-2])
        stack = stack[:-2] + [stack[-1]]

    while True:
        if len(stack) > 1 and stack[-1]["parent"] == stack[-2]["index"] and remain_childs[stack[-1]["index"]] == 0:
            arc_right()
        elif len(stack) > 1 and stack[-2]["parent"] == stack[-1]["index"] and remain_childs[stack[-2]["index"]] == 0:
            arc_left()
        elif len(buffer) >= 1:
            shift()
        else:
            if len(stack) > 1:
                nonprojective += 1
                configs = []
            break

    return configs

def print_configs(configs):
    for config in configs:
        print(' '.join(token['lemma'] for token in config[0]))
        print(' '.join(str(token['index']) for token in config[0]))
        print(' '.join(str(token['parent']) for token in config[0]))
        print(' '.join(str(token['left_children'][0]['index']) if len(token['left_children']) else "_" for token in config[0]))
        print(' '.join(str(token['right_children'][-1]['index']) if len(token['right_children']) else "_"  for token in config[0]))
        print(' '.join(token['lemma'] for token in config[1]))
        print(config[2])

def generate(configs):
    outputs = []
    for config in configs:
        outputs.append(generate_step(config))

    return outputs


def generate_step(config):
    stack, buffer, arc, relation = config
    sw = [] # feature of words
    st = [] # feature of pos
    sl = [] # feature of arc labels
    
    # s1 - sn
    for idx in range(1, TOP_N+1):
        sw.append(stack[-idx]['lemma'] if len(stack) >= idx else NULL_symbol)
        st.append(stack[-idx]['pos_tag'] if len(stack) >= idx else POS_prefix+NULL_symbol)
    # b1 - bn
    for idx in range(1, TOP_N+1):
        sw.append(buffer[idx-1]['lemma'] if len(buffer) >= idx else NULL_symbol)
        st.append(buffer[idx-1]['pos_tag'] if len(buffer) >= idx else POS_prefix+NULL_symbol)
    # lc1(si) - lcn(si), rc1(si) - rcn(si)
    for idx in range(1, TOP_N_C+1):
        for jdx in range(1, TOP_N_C_N+1):
            # lc1(si) - lcn(si)
            sw.append(stack[-idx]['left_children'][-jdx]['lemma'] if len(stack) >= idx and len(stack[-idx]['left_children']) >= jdx else NULL_symbol)
            st.append(stack[-idx]['left_children'][-jdx]['pos_tag'] if len(stack) >= idx and len(stack[-idx]['left_children']) >= jdx else POS_prefix+NULL_symbol)
            sl.append(stack[-idx]['left_children'][-jdx]['relation'] if len(stack) >= idx and len(stack[-idx]['left_children']) >= jdx else LABEL_prefix+NULL_symbol)
            # rc1(si) - rcn(si)
            sw.append(stack[-idx]['right_children'][-jdx]['lemma'] if len(stack) >= idx and len(stack[-idx]['right_children']) >= jdx else NULL_symbol)
            st.append(stack[-idx]['right_children'][-jdx]['pos_tag'] if len(stack) >= idx and len(stack[-idx]['right_children']) >= jdx else POS_prefix+NULL_symbol)
            sl.append(stack[-idx]['right_children'][-jdx]['relation'] if len(stack) >= idx and len(stack[-idx]['right_children']) >= jdx else LABEL_prefix+NULL_symbol)

    # lc(lc(si)), rc(rc(si))
    for idx in range(1, TOP_N_C+1):
        for direction in ['left_children', 'right_children']:
            sw.append(stack[-idx][direction][-1][direction][-1]['lemma'] if len(stack) >= idx and len(stack[-idx][direction]) and len(stack[-idx][direction][-1][direction]) else NULL_symbol)
            st.append(stack[-idx][direction][-1][direction][-1]['pos_tag'] if len(stack) >= idx and len(stack[-idx][direction]) and len(stack[-idx][direction][-1][direction]) else POS_prefix+NULL_symbol)
            sl.append(stack[-idx][direction][-1][direction][-1]['relation'] if len(stack) >= idx and len(stack[-idx][direction]) and len(stack[-idx][direction][-1][direction]) else LABEL_prefix+NULL_symbol)

    return [sw, st, sl, arc, relation]

def get_outputs(path, subset):
    samples = read_conll(path)
    samples_stat(samples)
    configs = []
    for sample in tqdm(samples):
        configs.extend(create_config(sample))

    outputs = generate(configs)
    pickle.dump(outputs, open(subset+'.converted', 'wb'))

if __name__ == "__main__":
    path = sys.argv[1]
    subset = sys.argv[2]

    get_outputs(path, subset)
    
    