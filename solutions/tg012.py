# -*- coding: utf-8 -*-
"""
Grupo tg012
Guilherme Saraiva #93717
Sara Ferreira #93756
"""
from collections import Counter
import numpy as np
from itertools import groupby
from math import log


# --------------------------------
#       Auxiliar functions
# --------------------------------

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def most_frequent(lst):
    data = Counter(lst)
    return max(lst, key=data.get)


def get_column(exs, idx):
    return [row[idx] for row in exs]


def entropy(s):
    if 0 in s:
        return 0

    sum_s = sum(s) * 1.0
    frac_t = s[0] / sum_s
    frac_f = s[1] / sum_s

    return -(frac_t * log(frac_t, 2) + frac_f * log(frac_f, 2))


def gain(ex, cfn):
    t = [0, 0]
    f = [0, 0]

    for i in range(len(ex)):
        if ex[i] == 0:
            t[cfn[i]] += 1
        else:
            f[cfn[i]] += 1

    s = [t[i] + f[i] for i in range(len(t))]
    sum_t = sum(t) * 1.0
    sum_f = sum(f) * 1.0
    sum_s = sum(s) * 1.0

    return entropy(s) - (sum_t / sum_s * entropy(t) + sum_f / sum_s * entropy(f))


def choose_attribute(examples, attributes, cfn):
    gains = [gain(get_column(examples, at), cfn) for at in attributes]
    idx = gains.index(max(gains))
    return attributes[idx]


# --------------------------------
#          Main functions
# --------------------------------

def dtl(examples, attributes, prev_cfn, noise):

    cfn = get_column(examples, -1)  # classification

    if not examples:
        return most_frequent(prev_cfn)

    elif all_equal(cfn):
        return cfn[0]

    elif noise and not attributes:
        return most_frequent(cfn)

    else:
        best = choose_attribute(examples, attributes, cfn)  # best statistically attribute
        tree = [best]                                       # put best on tree root

        for v in (0, 1):                                    # for each possible value in best column
            exs = [e for e in examples if e[best] == v]     # lines that contains value in best column
            new_attr = attributes.copy()                    # save attributes for each branch
            new_attr.remove(best)                           # remove best from attributes
            subtree = dtl(exs, new_attr, cfn, noise)        # recursively grows the subtree
            tree.append(subtree)                            # add subtree to tree

        return tree


def pruning(tree):
    if isinstance(tree[1], list) and isinstance(tree[2], list) and tree[1] == tree[2]:
        return pruning(tree[1])

    elif isinstance(tree[1], int) and isinstance(tree[2], list):
        tree[2] = pruning(tree[2])
        return tree

    elif isinstance(tree[1], list) and isinstance(tree[2], int):
        tree[1] = pruning(tree[1])
        return tree

    elif isinstance(tree[1], int) and isinstance(tree[2], int):
        return tree

    else:
        tree[1] = pruning(tree[1])
        tree[2] = pruning(tree[2])
        return tree


def pruning2(tree):
    if isinstance(tree[1], list) and isinstance(tree[2], list) and tree[1][0] == tree[2][0]:

        if tree[1][1] == tree[2][1]:
            tree = [tree[1][0], tree[1][1], [tree[0], tree[1][2], tree[2][2]]]
            return pruning2(tree)

        elif tree[1][2] == tree[2][2]:
            tree = [tree[1][0], [tree[0], tree[1][1], tree[2][1]], tree[1][2]]
            return pruning2(tree)

        else:
            tree[1] = pruning2(tree[1])
            tree[2] = pruning2(tree[2])
            return tree

    elif isinstance(tree[1], list) and isinstance(tree[2], int):
        tree[1] = pruning2(tree[1])
        return tree

    elif isinstance(tree[1], int) and isinstance(tree[2], list):
        tree[2] = pruning2(tree[2])
        return tree

    elif isinstance(tree[1], int) and isinstance(tree[2], int):
        return tree

    else:
        tree[1] = pruning2(tree[1])
        tree[2] = pruning2(tree[2])
        return tree


def createdecisiontree(D, Y, noise=False):

    if all_equal(Y):                                # check if all examples have got the same classification
        return [0, int(Y[0]), int(Y[0])]

    examples = np.c_[D, Y].astype(int).tolist()     # each examples line will have its classification
    attributes = list(range(0, len(D[0])))          # create attributes for each examples column except classification

    tree = dtl(examples, attributes, [], noise)     # get decision tree
    tree = pruning(tree)                            # prune equal childs
    tree = pruning2(tree)                           # prune attributes equal childs
    return tree

