import json
import preprocess as pp
import project_utils as utils

import numpy as np
from collections import defaultdict

threshold_beam = 2


"""
viterbi algorithm for inference max probability and corresponding backpointers 
"""


def memm_viterbi(tags, sentence, weights, feature_to_id):

    history_mechane_dict = {}
    n = len(sentence.split(' '))
    pi = {}
    bp = {}

    k = -1
    u = '*'
    v = '*'

    pi[k] = {}
    pi[k][u] = {}
    pi[k][u][v] = 1

    bp[k] = {}
    bp[k][u] = {}
    bp[k][u][v] = '*'

    sk = tags

    k += 1
    pi[k] = {}
    bp[k] = {}

    for v in sk:
        pi, bp, history_mechane_dict = update_pi_and_bp(0, '*', v, pi, tags, weights, sentence, feature_to_id, bp,
                                                        history_mechane_dict)


    top_beam_pi, top_beam_bp = get_top_k_tags(pi, k, threshold_beam, bp)
    pi, bp = update_pi_and_bp_after_beam(pi, top_beam_pi, k, bp, top_beam_bp)

    for k in range(1, n):
        pi[k] = {}
        bp[k] = {}
        for u in sk:
            for v in sk:
                pi, bp, history_mechane_dict = update_pi_and_bp(k, u, v, pi, tags, weights, sentence, feature_to_id, bp,
                                                                history_mechane_dict)

        top_beam_pi, top_beam_bp = get_top_k_tags(pi, k, threshold_beam, bp)
        pi, bp = update_pi_and_bp_after_beam(pi, top_beam_pi, k, bp, top_beam_bp)

    t = back_step(pi, bp, k)
    return t, pi, bp


"""
after forward pass of viterbi, make the backward pass and find the tags that maximizes the probability
"""


def back_step(pi, bp, k):
    last_pi = pi[k]

    max_prob = 0.0
    max_u_v = ()
    for u, v_dict in last_pi.items():
        for v, prob in v_dict.items():
            if prob > max_prob:
                max_prob = prob
                max_u_v = (u, v)

    if k > 0:
        t = {k: max_u_v[1], k - 1: max_u_v[0]}
    else:
        t = {k: max_u_v[1]}

    for i in range(k - 2, -1, -1):
        t[i] = bp[i + 2][t[i + 1]][t[i + 2]]

    return t


"""
after one step in viterbi, keep only the top k max elements corresponds to the top k probabilities
"""


def update_pi_and_bp_after_beam(pi, top_beam_pi, k, bp, top_beam_bp):
    pi[k] = {}
    bp[k] = {}

    for u_v, max_prob in top_beam_pi.items():
        u = u_v[0]
        v = u_v[1]

        if u not in pi[k]:
            pi[k][u] = {}

        if u not in bp[k]:
            bp[k][u] = {}

        pi[k][u][v] = max_prob  # pi[k][u][v] = max_prob

        max_t = top_beam_bp[u_v]
        bp[k][u][v] = max_t  # bp[k][u][v] = max_t

    return pi, bp


"""
gets : dicts of the following shape - 
pi[k][u][v] = max probability 
bp[k][u][v] = t = argmax of this prob 
"""


def get_top_k_tags(pi, k, threshold_param, bp):
    u_dict = pi[k]
    bp_dict = bp[k]

    all_pi_values_dict = {}
    all_bp_values_dict = {}

    for u, v_dict in u_dict.items():
        for v, pi_k_u_v in v_dict.items():
            all_pi_values_dict[(u, v)] = pi_k_u_v  # pi_k_u_v = max_prob = pi[k][u][v]

    for u, v_dict in bp_dict.items():
        for v, bp_k_u_v in v_dict.items():
            all_bp_values_dict[(u, v)] = bp_k_u_v  # bp_k_u_v = max_t = bp[k][u][v]

    top_beam_pi = {key: value for key, value in all_pi_values_dict.items() if value in
                   sorted(set(all_pi_values_dict.values()), reverse=True)[:threshold_param]}  # u_v -> prob

    top_beam_bp = {}
    for u_v, prob in top_beam_pi.items():
        top_beam_bp[u_v] = all_bp_values_dict[u_v]  # u_v -> max_t

    return top_beam_pi, top_beam_bp


"""
calc probability of history and tag
"""


def calc_prob(v, u, t, sentence, k, weights, feature_to_id, mechane):
    mone = calc_exp(v, u, t, sentence, k, weights, feature_to_id)
    return mone/mechane


"""
calc sum of exponent terms
"""


def calc_exp(v, u, t, sentence, k, weights, feature_to_id):
    hist = pp.History(u, t, sentence, k)
    feature_vector = hist.create_feature_vec(feature_to_id, v)
    return np.exp(sum([weights[entry] for entry in feature_vector]))


"""
calc probability with the denominator object - in order to speed up optimization
"""


def calc_mechane_and_probe(hist, w, tags, feature_to_id, v):
    mechane = 0.0
    mone = 0.0

    for t in tags:
        feat_vec = hist.create_feature_vec(feature_to_id, t)
        exp_term = np.exp(sum([w[entry] for entry in feat_vec]))
        mechane += exp_term
        if v == t:
            mone = exp_term

    prob = mone/mechane
    return prob, mechane


"""
one step of viterbi algorithm, return max prob and max t
"""


def update_pi_and_bp(k, u, v, pi, tags, weights, sentence, feature_to_id, bp, history_mechane_dict):

    max_prob = 0.0
    max_t = ''

    if k <= 0:
        tags = ['*'] + tags

    tags_for_this_k = tags
    if k > 0:
        tags_for_this_k = pi[k - 1]

    for t in tags_for_this_k:

        if t not in pi[k - 1]:
            continue

        if u not in pi[k - 1][t]:
            continue

        else:
            x = pi[k - 1][t][u]
            if x == 0.0:
                continue

            hist = pp.History(u, t, sentence, k)
            hist_key = hist.t_minus_1 + '_' + hist.t_minus_2 + '_' + hist.sentence + '_' + str(hist.index)

            if hist_key not in history_mechane_dict:
                q, mechane = calc_mechane_and_probe(hist, weights, tags, feature_to_id, v)
                history_mechane_dict[hist_key] = mechane
            else:
                mechane = history_mechane_dict[hist_key]
                q = calc_prob(v, u, t, sentence, k, weights, feature_to_id, mechane)

            prob = x * q

            if prob > max_prob:
                max_prob = prob
                max_t = t

    if max_prob != 0:
        if u not in pi[k]:
            pi[k][u] = {}
            bp[k][u] = {}
        pi[k][u][v] = max_prob
        bp[k][u][v] = max_t

    return pi, bp, history_mechane_dict


"""
create confusion matrix from predicted tags 
"""


def create_confusion_matrix_and_return_accuracy(all_tags, ground_truth, pred):
    matrix = {}  # matrix[pred][actual]
    for tag in all_tags:
        matrix[tag] = defaultdict(int)

    for i, pred_tag in pred.items():
        if i == -1:
            continue
        matrix[pred_tag][ground_truth[i]] += 1

    accuracy = 0.0
    total = 0.0

    for tag1 in all_tags:
        accuracy += matrix[tag1][tag1]
        for tag2 in all_tags:
            total += matrix[tag1][tag2]

    if total == 0:
        return 0, matrix
    return accuracy / total, matrix


"""
calc accuracy of predicted tags
"""


def calc_accuracy_from_all_matrices(all_matrices):
    mat = all_matrices[0]

    for i in range(1, len(all_matrices)):
        cur_mat = all_matrices[i]
        for key, val_dict in cur_mat.items():
            for inner_key, val in val_dict.items():
                if inner_key not in mat[key]:
                    mat[key][inner_key] = val
                else:
                    mat[key][inner_key] += val

    accuracy = 0.0
    total = 0.0

    for tag1 in list(mat.keys()):
        accuracy += mat[tag1][tag1]
        for tag2 in list(mat.keys()):
            total += mat[tag1][tag2]

    return accuracy/total, mat


"""
the inference function - get data and weights and make prediction of tags
"""

def inference_viterbi(data, weights, all_tags, feature_to_id, beam_param=2):
    global threshold_beam
    threshold_beam = beam_param

    all_matrices = []
    all_acc = []

    for sample in data:
        sample = data[sample]
        sentence = sample['sentence']

        pred, pi, bp = memm_viterbi(all_tags, sentence, weights, feature_to_id)
        accuracy, matrix = create_confusion_matrix_and_return_accuracy(all_tags, sample['tags'], pred)
        all_acc.append(accuracy)
        all_matrices.append(matrix)

    acc, mat = calc_accuracy_from_all_matrices(all_matrices)
    return acc, mat


"""
the inference function - get data and weights and make prediction of tags for compatition 
"""


def inference_viterbi_comp(data, weights, all_tags, feature_to_id, beam_param=2):
    print('viterbi comp!')
    global threshold_beam
    threshold_beam = beam_param
    print('Threshold beam: {}'.format(threshold_beam))
    tagged_data = []
    j = 0
    for sample in data:
        j += 1
        sample = data[sample]
        sentence = sample['sentence']
        sentence_split = sample['sentence'].split(' ')

        pred, pi, bp = memm_viterbi(all_tags, sentence, weights, feature_to_id)
        pred_sorted = dict(sorted(pred.items(), key=lambda item: item[0]))
        with_tag = ''
        for i, tag in pred_sorted.items():
            with_tag = with_tag + ' ' + sentence_split[i] + '_' + tag
        with_tag = with_tag[1:]
        tagged_data.append(with_tag)

    return tagged_data
