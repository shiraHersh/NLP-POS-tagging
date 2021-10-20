import pickle
import json
import pandas as pd
import random

import preprocess as pp
import optimization as opt
import inference as inf
import project_utils as utils

cut_num = -1

"""
split data for cross validation
"""

def split_data(path, p=0.2, data_index=1):
    data = []
    with open(path) as f:
        for line in f:
            if '\n' == line[-1]:
                line = line[:-1]
            data.append(line)

    random.shuffle(data)
    data_len = len(data)
    val = data[:int(data_len*p)]
    train = data[int(data_len*p):]

    with open(utils.data_path + "/train2_train" + str(data_index) + ".wtag", "w") as f:
        for sentence in train:
            sentence += "\n"
            f.write(sentence)

    with open(utils.data_path + "/train2_val" + str(data_index) + ".wtag", "w") as f:
        for sentence in val:
            sentence += "\n"
            f.write(sentence)

    print('p: {}\ntotal len: {}\ntrain len: {}\nval len: {}'.format(p, data_len, len(train), len(val)))


"""
set all values of threshold dict to specific value
"""


def set_one_val(d, val):
    new_d = {}
    for x, y in d.items():
        new_d[x] = val
    return new_d


"""
save the threshold dict and the accuracy achieved using it
"""


def save_threshold_dict_and_acc(threshold_dict, acc):
    new_key = ''
    d = {}
    for key, val in threshold_dict.items():
        new_key = new_key + '_' + str(key) + '_' + str(val)
    d[new_key] = acc

    with open('total_measures_dicts_.json', 'w') as f:
        json.dump(d, f)


"""
train model and evaluate on the same data
"""


def train_and_infer(mode='train1', beam_param=3, lmbda=2.0):

    if '1' in mode:
        threshold_dict_percentage = utils.best_model_1_percentages
    else:
        threshold_dict_percentage = utils.best_model_2_percentages

    weights_path = utils.trained_model_main_path + '/' + mode + '_model.pkl'

    data_obj = pp.preprocess_training_data(threshold_dict_percentage, cut_num=cut_num, mode=mode)
    weights = opt.optimize(data_obj, mode, lmbda)
    with open(weights_path, 'wb') as f:
        pickle.dump(weights, f)

    acc, mat = evaluation(learned_on=mode, test_on=mode, beam_param=beam_param)
    save_threshold_dict_and_acc(threshold_dict_percentage, acc)
    return acc, mat, weights


"""
evaluate data 'test_on' using model trained on parameter 'learned_on'
"""


def evaluation(learned_on='train1',
               test_on='test1',
               beam_param=3):

    print("\nEvaluating data {} from model learned on {}".format(test_on, learned_on))
    print('______________________________________________')

    weights_path = utils.trained_model_main_path + '/' + learned_on + '_model.pkl'

    print('weights path: {}'.format(weights_path))

    with open(utils.json_main_path + '/feature_to_id_' + learned_on + '.json', 'r') as f:
        feature_to_id = json.load(f)

    with open(utils.json_main_path + '/tags.json', 'r') as f:
        all_tags = json.load(f)

    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)

    acc = 0.0
    mat = {}

    if 'comp' in test_on:
        path = utils.data_path + '/' + test_on + '.words'
        data = pp.preprocess_comp_data(path)
        tagged_data = inf.inference_viterbi_comp(data, weights, all_tags, feature_to_id, beam_param=beam_param)

        with open(utils.data_path + '/' + 'comp_m' + test_on[-1] + '_311545818.wtag', 'w') as f:
            for sample in tagged_data:
                f.write(sample + '\n')

    else:
        path = utils.data_path + '/' + test_on + '.wtag'
        data = pp.preprocess_eval_data(all_tags, feature_to_id, path)
        acc, mat = inf.inference_viterbi(data, weights, all_tags, feature_to_id, beam_param=beam_param)
        print("\n**************\nThe inference accuracy for mode {} is: {:.4f}\n*************\n".format(test_on, acc))
        pd.DataFrame(mat).to_csv(test_on + '_confusion_matrix.csv')

    return acc, mat


"""
cross validation for model 2
"""


def cross_validation(p=0.2, mode='train2', model_based='train2_train'):
    print("Starting CV")
    best_train_score = 0.0
    best_val_score = 0.0
    weights_with_acc = {}
    best_val_index = 0
    for i in range(5):
        split_data(utils.data_path + '/' + mode + '.wtag', p, i)
        tr_acc, tr_mat, weights = train_and_infer(mode=model_based + str(i), lmbda=0.75)
        val_acc, val_mat = evaluation(learned_on='train2_train' + str(i), test_on='train2_val' + str(i))
        print('i: {} , tr_acc : {} , validation accuracy: {}'.format(i, tr_acc, val_acc))
        if val_acc > best_val_score:
            best_val_index = i
            best_val_score = val_acc
            best_train_score = tr_acc

        weights_with_acc[i] = {'weights': weights, 'train acc': tr_acc, 'val acc': val_acc}

    weights_path = utils.trained_model_main_path + '/' + 'train2_train' + '_model.pkl'
    with open(weights_path, 'wb') as f:
        pickle.dump(weights_with_acc[best_val_index], f)

    print('\nCross validation best val accuracy: \nTrain: {:.3f}\nVal: {:.3f}\n'.
          format(best_train_score, best_val_score))
    return weights_with_acc, best_val_index


# def check_different_threshold_dicts(learn_on='train2', check_on='test1'):
#     from copy import deepcopy
#     original_dict = deepcopy(utils.checking_percentages)
#     best_acc_for_eval = 0.0
#     best_key_to_shut_down = ''
#
#     for key, thresh in original_dict.items():
#         utils.checking_percentages = deepcopy(original_dict)
#         utils.checking_percentages[key] = 0
#         print('\n------------- Key : {} is now 0 -------------\n'.format(key))
#         train_and_infer(mode=learn_on, beam_param=3, lmbda=0.5, checking_threshold_dict=True)
#         acc, _ = evaluation(learned_on=learn_on, test_on=check_on, beam_param=3)
#         if acc > best_acc_for_eval:
#             best_acc_for_eval = acc
#             best_key_to_shut_down = key
#
#     print("\n\n Conclusions: \nBest " + check_on + " Accuracy: {}, when shutting down key: {}".format(
#                                                             best_acc_for_eval, best_key_to_shut_down))


# # Model 1
# train_and_infer(mode='train1', lmbda=2)
# acc, mat = evaluation(learned_on='train1', test_on='test1')
# pd.DataFrame(mat).to_csv('confusion_matrix_test1.csv')
# print(mat)
# evaluation(learned_on='train1', test_on='comp1', beam_param=5)
#
# best_test_acc = 0
# best_lambda_acc = 0
# for lam in [0, 0.15, 0.25]:
#     # Model 2 - without CrossValidation
#     train_and_infer(mode='train2', beam_param=3, lmbda=lam)
#     accuracy, matrix = evaluation(learned_on='train2', test_on='test1')
#     if accuracy > best_test_acc:
#         best_test_acc = accuracy
#         best_lambda_acc = lam
#     evaluation(learned_on='train2', test_on='comp2', beam_param=3)
#
# print('best_test_acc: {}, best_lambda_acc: {}'.format(best_test_acc, best_lambda_acc))
# Model 2 - with CrossValidation
# weights_with_acc, best_val_index = cross_validation()
# evaluation(learned_on='train2_train' + str(best_val_index), test_on='train2_val' + str(best_val_index))
# evaluation(learned_on='train2_train' + str(best_val_index), test_on='comp2')


# check_different_threshold_dicts()
