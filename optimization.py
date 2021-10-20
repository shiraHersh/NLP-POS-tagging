import json
import preprocess as pp
import numpy as np
import pickle
from scipy.optimize import fmin_l_bfgs_b
import project_utils as utils


"""
calculate empirical count for optimization equation 
"""


# Read preprocessed data
def create_empirical_count_from_feature_counter(feature_counter, num_of_features):
    empirical_count = np.zeros(num_of_features)
    for feature_id, count in feature_counter.items():
        empirical_count[int(feature_id)] = count

    return empirical_count


"""
read preprocessed json files  and convert it to data object
"""


def read_json_and_convert_to_data(mode, data_from_preprocess):
    if data_from_preprocess != {}:
        json_obj = data_from_preprocess['preprocessed_data']
        num_of_features = data_from_preprocess['number_of_features']
        feature_counter = data_from_preprocess['feature_counter']

    else:
        with open(utils.json_main_path + '/preprocessed_data_' + mode + '.json', 'r') as f:
            json_obj = json.load(f)

        with open(utils.json_main_path + '/number_of_features_' + mode + '.json', 'r') as f:
            num_of_features = json.load(f)

        with open(utils.json_main_path + '/feature_counter_' + mode + '.json', 'r') as f:
            feature_counter = json.load(f)

    empirical_count = create_empirical_count_from_feature_counter(feature_counter, num_of_features)

    data = pp.from_js_to_data(json_obj)

    return data, num_of_features, empirical_count


"""
calc inner linear term - v*f(h,t)
"""


# Linear term
def inner_linear_term(sample, v, word, tag):
    feature_obj = sample['feature_vectors']
    vec = feature_obj[word][tag]
    return sum([v[entry] for entry in vec])


"""
calc linear term of one sample - sigma(v*f(h,t))
"""


def calc_linear_term_of_one_sample(sample, v_t):
    linear_term = 0
    feature_obj = sample['feature_vectors']
    sentence_in_list = list(feature_obj.keys())

    for i in range(len(sentence_in_list)):
        word = sentence_in_list[i]
        tag = sample['tags'][i]
        linear_term += inner_linear_term(sample, v_t, word, tag)

    return linear_term


"""
calc exponent term of all tags for the denominator
"""


# normalization term
def calc_exp_term_for_all_tags(sample, v_t, word):
    all_tags_linear_term = 0
    feature_obj = sample['feature_vectors']

    for tag in feature_obj[word]:
        all_tags_linear_term += np.exp(inner_linear_term(sample, v_t, word, tag))

    return all_tags_linear_term


"""
calc normalization term of one sample in the data
"""


def calc_normalization_term_of_one_sample(sample, v_t, expected_counts_mechane):
    normalization_term = 0
    feature_obj = sample['feature_vectors']
    sentence_in_list = list(feature_obj.keys())

    for i in range(len(sentence_in_list)):
        word = sentence_in_list[i]
        sum_for_this_word = calc_exp_term_for_all_tags(sample, v_t, word)
        expected_counts_mechane[sample['id']][word] = sum_for_this_word
        normalization_term += np.log(sum_for_this_word)

    return normalization_term, expected_counts_mechane


"""
calc expected counts of one sample in the data
"""


# calculate likelihood
def calc_expected_count_of_one_sample(sample, v_t, expected_counts_mechane, sum_of_vectors):
    feature_obj = sample['feature_vectors']
    sentence_in_list = list(feature_obj.keys())

    for i in range(len(sentence_in_list)):
        word = sentence_in_list[i]
        for tag in feature_obj[word]:
            prob = calc_prob(sample, word, tag, v_t, expected_counts_mechane)

            entries = feature_obj[word][tag]
            for entry in entries:
                sum_of_vectors[entry] += prob

    return sum_of_vectors


"""
calc probability of specific history and tag
"""


def calc_prob(sample, word, tag, v_t, expected_counts_mechane):
    mone = np.exp(inner_linear_term(sample, v_t, word, tag))
    mechane = expected_counts_mechane[sample['id']][word]

    return mone / mechane


counter = 0
likelihood = 0
grad = 0


"""
calc grad and likelihood of one iteration
"""


def calc_objective_per_iter(v_t, data, lmbda, num_of_features, empirical_counts):
    global counter
    global likelihood
    global grad
    counter += 1

    norm_term = 0
    regularization_grad = lmbda * v_t

    v_norm = np.linalg.norm(v_t)
    regularization = 0.5 * lmbda * (v_norm ** 2)

    expected_counts_mechane = {}
    expected_counts = np.zeros(num_of_features)

    total_linear_term = np.inner(empirical_counts, v_t)

    for sample_id in data:
        sample = data[sample_id]
        expected_counts_mechane[int(sample_id)] = {}

        norm, expected_counts_mechane = calc_normalization_term_of_one_sample(sample, v_t, expected_counts_mechane)
        norm_term += norm

        expected_counts = calc_expected_count_of_one_sample(sample, v_t, expected_counts_mechane, expected_counts)

    likelihood = total_linear_term - norm_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    if counter % 10 == 0:
        print('iteration: {} ,likelihood: {:.5f}'.format(counter, likelihood))

    return (-1) * likelihood, (-1) * grad


"""
read data object, and find optimal weights vectors that maximizes the likelihood 
"""


def optimize(data_from_preprocess, mode='', lmbda=2):
    global counter

    data, num_of_features, empirical_count = read_json_and_convert_to_data(mode, data_from_preprocess)

    args = (data, lmbda, num_of_features, empirical_count)
    w_0 = np.random.rand(num_of_features)
    counter = 0
    optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=0)

    weights = optimal_params[0]

    # weights_path = utils.trained_model_main_path + '/trained_weights_' + mode + '.pkl'
    # with open(weights_path, 'wb') as f:
    #     pickle.dump(weights, f)

    return weights
