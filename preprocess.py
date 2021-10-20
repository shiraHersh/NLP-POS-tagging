from collections import defaultdict
import project_utils as utils
import json

path = utils.data_path + '/train2.wtag'
json_path = utils.json_main_path
word_shape_dict = {}
threshold_dict_percentage = utils.best_model_2_percentages

"""
FeatureStatistics - class of features and their appearance statistics
"""


class FeatureStatistics:

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        # Basic features
        self.words_count_dict = defaultdict(int)
        self.tags_count_dict = defaultdict(int)

        self.words_tags_count_dict = defaultdict(int)

        self.bigram_tags_count_dict = defaultdict(int)
        self.trigram_tags_count_dict = defaultdict(int)

        # Advanced features
        self.t_i_w_i_suffix_len_4 = defaultdict(int)
        self.t_i_w_i_prefix_len_4 = defaultdict(int)
        self.t_i_w_i_suffix_len_3 = defaultdict(int)
        self.t_i_w_i_prefix_len_3 = defaultdict(int)
        self.t_i_w_i_suffix_len_2 = defaultdict(int)
        self.t_i_w_i_prefix_len_2 = defaultdict(int)
        self.t_i_w_i_suffix_len_1 = defaultdict(int)
        self.t_i_w_i_prefix_len_1 = defaultdict(int)

        self.t_i_w_contains_capital_dict = defaultdict(int)
        self.t_i_w_contains_number_dict = defaultdict(int)

        self.t_w_full_shape_dict = defaultdict(int)
        self.t_w_small_shape_dict = defaultdict(int)

        self.ti_wi_minus1 = defaultdict(int)
        self.ti_wi_plus1 = defaultdict(int)

        # Additional features for data 2
        self.ti_first_letter_capitalized = defaultdict(int)
        self.ti_first_word_in_sentence = defaultdict(int)
        self.ti_last_word_in_sentence = defaultdict(int)
        self.ti_all_cap = defaultdict(int)
        self.ti_all_small = defaultdict(int)
        self.ti_has_hyphen = defaultdict(int)
        
        # checking
        self.ti_wi_minus2_wi_minus_1_wi = defaultdict(int)
        self.ti_wi_minus2_wi_minus_1 = defaultdict(int)
        self.ti_wi_minus_1_wi = defaultdict(int)

    """
    functions for features retrieval from data
    """


    def get_features_from_data(self, file_path):
        with open(file_path) as f:
            for line in f:

                if "\n" == line[-1]:
                    line = line[:-1]
                split_words = line.split(' ')

                self.get_word_tag_pair_count(split_words)
                self.get_bigram_trigram_tag_pair_count(split_words)
                self.get_advanced_contain_features(split_words)
                self.get_advanced_shape_features(split_words)
                self.get_advanced_bigram_trigram_tag_pair_count(split_words)
                self.get_advanced_features_for_data2(split_words)

    def get_word_tag_pair_count(self, split_words):
        for word_idx in range(len(split_words)):
            cur_word, cur_tag = split_words[word_idx].split('_')
            self.words_tags_count_dict[(cur_word, cur_tag)] += 1
            self.words_count_dict[cur_word] += 1
            self.tags_count_dict[cur_tag] += 1

            if len(cur_word) >= 4:
                self.t_i_w_i_suffix_len_4[(cur_tag, cur_word[-4:])] += 1
                self.t_i_w_i_prefix_len_4[(cur_tag, cur_word[:4])] += 1
                self.t_i_w_i_suffix_len_3[(cur_tag, cur_word[-3:])] += 1
                self.t_i_w_i_prefix_len_3[(cur_tag, cur_word[:3])] += 1
                self.t_i_w_i_suffix_len_2[(cur_tag, cur_word[-2:])] += 1
                self.t_i_w_i_prefix_len_2[(cur_tag, cur_word[:2])] += 1
                self.t_i_w_i_suffix_len_1[(cur_tag, cur_word[-1:])] += 1
                self.t_i_w_i_prefix_len_1[(cur_tag, cur_word[:1])] += 1

    def get_bigram_trigram_tag_pair_count(self, split_words):
        for i in range(1, len(split_words)):
            tag = split_words[i].split('_')[1]
            tag_minus_1 = split_words[i - 1].split('_')[1]
            
            w = split_words[i].split('_')[0]
            w_minus_1 = split_words[i - 1].split('_')[0]

            if i == 1:
                self.bigram_tags_count_dict[('*', tag_minus_1)] += 1

                self.trigram_tags_count_dict[('*', '*', tag_minus_1)] += 1
                self.trigram_tags_count_dict[('*', tag_minus_1, tag)] += 1

                self.ti_wi_minus2_wi_minus_1_wi[(tag_minus_1, '', '', w_minus_1)] += 1
                self.ti_wi_minus2_wi_minus_1_wi[(tag, '', w_minus_1, w)] += 1
                self.ti_wi_minus2_wi_minus_1[(tag, '', w_minus_1)] += 1
                self.ti_wi_minus_1_wi[(tag_minus_1, '', w_minus_1)] += 1

            if i > 1:
                tag_minus_2 = split_words[i - 2].split('_')[1]
                w_minus_2 = split_words[i - 2].split('_')[0]
                self.trigram_tags_count_dict[(tag_minus_2, tag_minus_1, tag)] += 1
                
                self.ti_wi_minus2_wi_minus_1_wi[(tag, w_minus_2, w_minus_1, w)] += 1
                self.ti_wi_minus2_wi_minus_1_wi[(tag, w_minus_2, w_minus_1)] += 1
                self.ti_wi_minus2_wi_minus_1[(tag, w_minus_2, w_minus_1)] += 1

            if i == len(split_words) - 1:
                self.bigram_tags_count_dict[(tag, '._.')] += 1
                self.trigram_tags_count_dict[(tag_minus_1, tag, '._.')] += 1

            self.bigram_tags_count_dict[(tag_minus_1, tag)] += 1
            self.ti_wi_minus_1_wi[(tag, w_minus_1, w)] += 1

    def get_advanced_contain_features(self, split_words):
        for word_idx in range(len(split_words)):
            cur_word, cur_tag = split_words[word_idx].split('_')

            if utils.contain_num(cur_word):
                self.t_i_w_contains_number_dict[cur_tag] += 1

            if utils.contain_capital(cur_word):
                self.t_i_w_contains_capital_dict[cur_tag] += 1

    def get_advanced_shape_features(self, split_words):
        for word_idx in range(len(split_words)):
            cur_word, cur_tag = split_words[word_idx].split('_')
            word_full_shape, word_small_shape = utils.get_word_shape(cur_word, word_shape_dict)

            self.t_w_full_shape_dict[cur_tag, word_full_shape] += 1
            self.t_w_small_shape_dict[cur_tag, word_small_shape] += 1

    def get_advanced_bigram_trigram_tag_pair_count(self, split_words):

        # < tag_i, w_i+1 >
        # < tag_i, tag_i+1 >
        for i in range(len(split_words) - 1):
            tag_i = split_words[i].split('_')[1]
            w_i_plus_1, tag_i_plus_1 = split_words[i + 1].split('_')

            self.ti_wi_plus1[(tag_i, w_i_plus_1)] += 1

        # < tag_i, w_i-1 >
        for i in range(1, len(split_words)):
            tag_i = split_words[i].split('_')[1]
            w_i_minus_1 = split_words[i - 1].split('_')[0]

            self.ti_wi_minus1[(tag_i, w_i_minus_1)] += 1

            if i == len(split_words) - 1:
                w_i = split_words[i].split('_')[1]
                self.ti_wi_minus1[('._.', w_i)] += 1

    def get_advanced_features_for_data2(self, split_words):
        for word_idx in range(len(split_words)):
            cur_word, cur_tag = split_words[word_idx].split('_')

            if word_idx == 0:
                self.ti_first_word_in_sentence[cur_tag] += 1

            if word_idx == len(split_words) - 1:
                self.ti_last_word_in_sentence[cur_tag] += 1

            if cur_word[0] != cur_word[0].lower():
                self.ti_first_letter_capitalized[cur_tag] += 1

            all_cap, all_small = utils.check_all_cap_all_small(cur_word)

            if all_cap:
                self.ti_all_cap[cur_tag] += 1

            if all_small:
                self.ti_all_small[cur_tag] += 1

            if '-' in cur_word:
                self.ti_has_hyphen[cur_tag] += 1

"""
class for mapping features to ids and vice versa
"""


class Feature2ID:
    def __init__(self, feature_statistics):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts

        self.n_total_features = 0  # Total number of features accumulated

        self.threshold_dict_percentage = threshold_dict_percentage
        self.dicts = self.prune_class_dicts()

        self.feature_to_id = {}
        self.id_to_feature = {}

    """
    prune dicts - take only top k according to threshold
    """
    def prune_class_dicts(self):
        all_dicts = \
            {'tags_vocab': utils.get_top_k_percentage(self.feature_statistics.tags_count_dict,
                                                      self.threshold_dict_percentage['tags_vocab'],
                                                      'tags_vocab'),

             'words_tags': utils.get_top_k_percentage(self.feature_statistics.words_tags_count_dict,
                                                      self.threshold_dict_percentage['words_tags'],
                                                      'words_tags'),

             'bigram_tags': utils.get_top_k_percentage(self.feature_statistics.bigram_tags_count_dict,
                                                       self.threshold_dict_percentage['bigram_tags'],
                                                       'bigram_tags'),

             'trigram_tags': utils.get_top_k_percentage(self.feature_statistics.trigram_tags_count_dict,
                                                        self.threshold_dict_percentage['trigram_tags'],
                                                        'trigram_tags'),

             'ti_suffix_len_4': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_suffix_len_4,
                                                           self.threshold_dict_percentage['ti_suffix_len_4'],
                                                           'ti_suffix_len_4'),

             'ti_prefix_len_4': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_prefix_len_4,
                                                           self.threshold_dict_percentage['ti_suffix_len_4'],
                                                           'ti_suffix_len_4'),

             'ti_suffix_len_3': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_suffix_len_3,
                                                           self.threshold_dict_percentage['ti_suffix_len_3'],
                                                           'ti_suffix_len_3'),

             'ti_prefix_len_3': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_prefix_len_3,
                                                           self.threshold_dict_percentage['ti_prefix_len_3'],
                                                           'ti_prefix_len_3'),

             'ti_suffix_len_2': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_suffix_len_2,
                                                           self.threshold_dict_percentage['ti_suffix_len_2'],
                                                           'ti_suffix_len_2'),

             'ti_prefix_len_2': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_prefix_len_2,
                                                           self.threshold_dict_percentage['ti_prefix_len_2'],
                                                           'ti_prefix_len_2'),

             'ti_suffix_len_1': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_suffix_len_1,
                                                           self.threshold_dict_percentage['ti_suffix_len_1'],
                                                           'ti_suffix_len_1'),

             'ti_prefix_len_1': utils.get_top_k_percentage(self.feature_statistics.t_i_w_i_prefix_len_1,
                                                           self.threshold_dict_percentage['ti_prefix_len_1'],
                                                           'ti_prefix_len_1'),

             'ti_capital': utils.get_top_k_percentage(self.feature_statistics.t_i_w_contains_capital_dict,
                                                      self.threshold_dict_percentage['ti_capital'],
                                                      'ti_capital'),

             'ti_num': utils.get_top_k_percentage(self.feature_statistics.t_i_w_contains_number_dict,
                                                  self.threshold_dict_percentage['ti_num'],
                                                  'ti_num'),

             'ti_wi_full_shape': utils.get_top_k_percentage(self.feature_statistics.t_w_full_shape_dict,
                                                            self.threshold_dict_percentage['ti_wi_full_shape'],
                                                            'ti_wi_full_shape'),

             'ti_wi_small_shape': utils.get_top_k_percentage(self.feature_statistics.t_w_small_shape_dict,
                                                             self.threshold_dict_percentage['ti_wi_small_shape'],
                                                             'ti_wi_small_shape'),

             'ti_wi_minus1': utils.get_top_k_percentage(self.feature_statistics.ti_wi_minus1,
                                                        self.threshold_dict_percentage['ti_wi_minus1'],
                                                        'ti_wi_minus1'),

             'ti_wi_plus1': utils.get_top_k_percentage(self.feature_statistics.ti_wi_plus1,
                                                       self.threshold_dict_percentage['ti_wi_plus1'],
                                                       'ti_wi_plus1'),

             # Advanced features for data 2
             'ti_first_letter_capitalized':
                 utils.get_top_k_percentage(self.feature_statistics.ti_first_letter_capitalized,
                                            self.threshold_dict_percentage['ti_first_cap'],
                                            'ti_first_cap'),

             'ti_first_w': utils.get_top_k_percentage(self.feature_statistics.ti_first_word_in_sentence,
                                                      self.threshold_dict_percentage['ti_first_w'],
                                                      'ti_first_w'),

             'ti_last_w': utils.get_top_k_percentage(self.feature_statistics.ti_last_word_in_sentence,
                                                     self.threshold_dict_percentage['ti_last_w'],
                                                     'ti_last_w'),

             'ti_all_cap': utils.get_top_k_percentage(self.feature_statistics.ti_all_cap,
                                                      self.threshold_dict_percentage['ti_all_cap'],
                                                      'ti_all_cap'),

             'ti_all_small': utils.get_top_k_percentage(self.feature_statistics.ti_all_small,
                                                        self.threshold_dict_percentage['ti_all_small'],
                                                        'ti_all_small'),

             'ti_has_hyphen': utils.get_top_k_percentage(self.feature_statistics.ti_has_hyphen,
                                                         self.threshold_dict_percentage['ti_has_hyphen'],
                                                         'ti_has_hyphen'),

             # More

             'ti_wi_minus2_wi_minus_1_wi': utils.get_top_k_percentage(
                                                      self.feature_statistics.ti_wi_minus2_wi_minus_1_wi,
                                                      self.threshold_dict_percentage['ti_wi_minus2_wi_minus_1_wi'],
                                                      'ti_wi_minus2_wi_minus_1_wi'),

             'ti_wi_minus2_wi_minus_1': utils.get_top_k_percentage(
                                                        self.feature_statistics.ti_wi_minus2_wi_minus_1,
                                                        self.threshold_dict_percentage['ti_wi_minus2_wi_minus_1'],
                                                        'ti_wi_minus2_wi_minus_1'),

             'ti_wi_minus_1_wi': utils.get_top_k_percentage(
                                                         self.feature_statistics.ti_wi_minus_1_wi,
                                                         self.threshold_dict_percentage['ti_wi_minus_1_wi'],
                                                         'ti_wi_minus_1_wi'),

             }
        return all_dicts

    """
    create feature to id dict - each feature will be encoded according to the dict below
    """
    def create_feature_id_dict(self):
        feature_to_id = defaultdict(id)
        id_to_feature = defaultdict(str)

        name_to_type = {'tags_vocab': 't',
                        'words_tags': 'wt',
                        'bigram_tags': 'bg',
                        'trigram_tags': 'tg',
                        'ti_suffix_len_4': 'wsfx4',
                        'ti_prefix_len_4': 'wpre4',
                        'ti_suffix_len_3': 'wsfx3',
                        'ti_prefix_len_3': 'wpre3',
                        'ti_suffix_len_2': 'wsfx2',
                        'ti_prefix_len_2': 'wpre2',
                        'ti_suffix_len_1': 'wsfx1',
                        'ti_prefix_len_1': 'wpre1',
                        'ti_capital': 'cap',
                        'ti_num': 'num',
                        'ti_wi_full_shape': 'fshape',
                        'ti_wi_small_shape': 'sshape',
                        'ti_wi_minus1': 'ti_wim1',
                        'ti_wi_plus1': 'ti_wip1',
                        'ti_first_letter_capitalized': 'ti_w[0]_cap',
                        'ti_first_w': 'ti_first_w',
                        'ti_last_w': 'ti_last_w',
                        'ti_all_cap': 'ti_all_cap',
                        'ti_all_small': 'ti_all_small',
                        'ti_has_hyphen': 'ti_has_hyphen',
                        
                        'ti_wi_minus2_wi_minus_1_wi': 'ti_wi_minus2_wi_minus_1_wi',
                        'ti_wi_minus2_wi_minus_1': 'ti_wi_minus2_wi_minus_1',
                        'ti_wi_minus_1_wi': 'ti_wi_minus_1_wi'
                        }

        for d_name, d in self.dicts.items():
            for data in d:
                # feat = feature(name_to_type[d_name], data, self.n_total_features)
                if isinstance(data, tuple):
                    feat = name_to_type[d_name] + '_' + '_'.join(data)
                else:
                    feat = name_to_type[d_name] + '_' + data

                id_to_feature[self.n_total_features] = feat
                feature_to_id[feat] = self.n_total_features
                self.n_total_features += 1

        self.feature_to_id = feature_to_id
        self.id_to_feature = id_to_feature
        print("\nTotal number of features: {}\n".format(self.n_total_features))
        return feature_to_id, id_to_feature


"""
History class - encodes all data according to the history object presented in class
params:  t_minus_1, t_minus_2, sentence, index
"""

class History:
    def __init__(self, t_minus_1, t_minus_2, sentence, index):
        self.t_minus_1 = t_minus_1
        self.t_minus_2 = t_minus_2

        split_sentence = sentence.split(' ')

        if index < len(split_sentence) - 1:
            self.w_plus_1 = split_sentence[index + 1]
        else:
            self.w_plus_1 = ''

        if index > 0:
            self.w_minus_1 = split_sentence[index - 1]
        else:
            self.w_minus_1 = ''

        self.sentence = sentence
        self.index = index
        self.cur_w = split_sentence[index]

    """
    create features for this history, look for all the features that exist 
    in this history and find their corresponding indexes
    """

    def create_feature_vec(self, feature_to_id, t):

        feature_code_names = ['t', 'wt', 'bg', 'tg', 'wsfx4', 'wpre4', 'wsfx3', 'wpre3', 'wsfx2', 'wpre2', 'wsfx1',
                              'wpre1', 'cap', 'num', 'fshape', 'sshape', 'ti_wim1', 'ti_wip1', 'ti_tip1',
                              'ti_tip1_tip2', 'ti_tip1_tim1', 'ti_w[0]_cap', 'ti_first_w', 'ti_last_w', 'ti_all_cap',
                              'ti_all_small', 'ti_has_hyphen', 'ti_wi_minus2_wi_minus_1_wi', 'ti_wi_minus2_wi_minus_1',
                              'ti_wi_minus_1_wi']

        features_names_for_this_history = []

        for feature_name in feature_code_names:
            if feature_name == 't':
                features_names_for_this_history.append('t_' + t)

            elif feature_name == 'wt':
                features_names_for_this_history.append('wt_' + self.cur_w + '_' + t)

            elif feature_name == 'bg':
                features_names_for_this_history.append('bg_' + self.t_minus_1 + '_' + t)

            elif feature_name == 'tg':
                features_names_for_this_history.append('tg_' + self.t_minus_2 + '_' + self.t_minus_1 + '_' + t)

            elif feature_name == 'cap':
                if utils.contain_capital(self.cur_w):
                    features_names_for_this_history.append('cap_' + t)

            elif feature_name == 'num':
                if utils.contain_num(self.cur_w):
                    features_names_for_this_history.append('num_' + t)

            elif feature_name == 'fshape':
                full_shape, _ = utils.get_word_shape(self.cur_w, word_shape_dict)
                features_names_for_this_history.append('fshape_' + t + '_' + full_shape)

            elif feature_name == 'sshape':
                _, small_shape = utils.get_word_shape(self.cur_w, word_shape_dict)
                features_names_for_this_history.append('sshape_' + t + '_' + small_shape)

            elif feature_name == 'ti_wim1':
                features_names_for_this_history.append('ti_wim1_' + t + '_' + self.w_minus_1)

            elif feature_name == 'ti_wip1':
                features_names_for_this_history.append('ti_wip1_' + t + '_' + self.w_plus_1)

            elif feature_name == 'ti_w[0]_cap':
                if self.cur_w[0] != self.cur_w[0].lower():
                    features_names_for_this_history.append('ti_w[0]_cap_' + t)

            elif feature_name == 'ti_first_w':
                if self.index == 0:
                    features_names_for_this_history.append('ti_first_w_' + t)

            elif feature_name == 'ti_last_w':
                if self.index == len(self.sentence.split(' ')) - 1:
                    features_names_for_this_history.append('ti_last_w_' + t)

            elif feature_name == 'ti_all_cap':
                all_cap, _ = utils.check_all_cap_all_small(self.cur_w)
                if all_cap:
                    features_names_for_this_history.append('ti_all_cap_' + t)

            elif feature_name == 'ti_all_small':
                _, all_small = utils.check_all_cap_all_small(self.cur_w)
                if all_small:
                    features_names_for_this_history.append('ti_all_small_' + t)

            elif feature_name == 'ti_has_hyphen':
                if '-' in self.cur_w:
                    features_names_for_this_history.append('ti_has_hyphen_' + t)

            elif feature_name == 'ti_wi_minus2_wi_minus_1_wi':
                w_minus_2 = ''
                w_minus_1 = ''
                w = self.cur_w
                if self.index > 1:
                    w_minus_2 = self.sentence[self.index - 2]
                if self.index > 0:
                    w_minus_1 = self.sentence[self.index - 1]
                features_names_for_this_history.append('ti_wi_minus2_wi_minus_1_wi_' +
                                                       w_minus_2 + '_' + w_minus_1 + '_' + w + '_' + t)

            elif feature_name == 'ti_wi_minus2_wi_minus_1':
                w_minus_2 = ''
                w_minus_1 = ''
                if self.index > 1:
                    w_minus_2 = self.sentence[self.index - 2]
                if self.index > 0:
                    w_minus_1 = self.sentence[self.index - 1]
                features_names_for_this_history.append('ti_wi_minus2_wi_minus_1_wi_' +
                                                       w_minus_2 + '_' + w_minus_1 + '_' + t)

            elif feature_name == 'ti_wi_minus_1_wi':
                w_minus_1 = ''
                w = self.cur_w
                if self.index > 0:
                    w_minus_1 = self.sentence[self.index - 1]
                features_names_for_this_history.append('ti_wi_minus2_wi_minus_1_wi_' + w_minus_1 + '_' + w + '_' + t)

            elif 'w' in feature_name and len(self.cur_w) >= 4:
                if feature_name == 'wsfx4':
                    features_names_for_this_history.append('wsfx4_' + t + '_' + self.cur_w[-4:])

                elif feature_name == 'wpre4':
                    features_names_for_this_history.append('wpre4_' + t + '_' + self.cur_w[:4])

                elif feature_name == 'wsfx3':
                    features_names_for_this_history.append('wsfx3_' + t + '_' + self.cur_w[-3:])

                elif feature_name == 'wpre3':
                    features_names_for_this_history.append('wpre3_' + t + '_' + self.cur_w[:3])

                elif feature_name == 'wsfx2':
                    features_names_for_this_history.append('wsfx2_' + t + '_' + self.cur_w[-2:])

                elif feature_name == 'wpre2':
                    features_names_for_this_history.append('wpre3_' + t + '_' + self.cur_w[:2])

                elif feature_name == 'wsfx1':
                    features_names_for_this_history.append('wsfx1_' + t + '_' + self.cur_w[-1:])

                elif feature_name == 'wpre1':
                    features_names_for_this_history.append('wpre3_' + t + '_' + self.cur_w[:1])

        vec = [feature_to_id[name] for name in features_names_for_this_history if name in feature_to_id]
        return vec


"""
create data object after preprocess for optimization 
"""


def create_data_after_preprocess(stats, feature_to_id):
    data = {}
    feature_counter = defaultdict(int)
    all_histories = defaultdict(dict)
    with open(path) as f:
        i = -1
        for line in f:
            i += 1

            if "\n" == line[-1]:
                line = line[:-1]

            splited_words = line.split(' ')

            sentence = ''
            tags = []
            words = []

            for word_idx in range(len(splited_words)):
                cur_word, cur_tag = splited_words[word_idx].split('_')
                sentence += ' ' + cur_word
                tags.append(cur_tag)
                words.append(cur_word)

            sentence = sentence[1:]
            sample = {'id': i, 'sentence': sentence, 'tags': tags, 'histories': {}, 'feature_vectors': {}}

            for word_idx in range(len(splited_words)):
                sample['histories'][words[word_idx]] = {}

                if word_idx == 0:
                    t_minus_1 = '*'
                    t_minus_2 = '*'

                elif word_idx == 1:
                    t_minus_1 = tags[word_idx - 1]
                    t_minus_2 = '*'

                else:
                    t_minus_1 = tags[word_idx - 1]
                    t_minus_2 = tags[word_idx - 2]

                hist = History(t_minus_1, t_minus_2, sentence, word_idx)
                sample['histories'][words[word_idx]] = hist

                all_histories[hist] = {}

                for t in list(stats.tags_count_dict.keys()):
                    feature_vec = hist.create_feature_vec(feature_to_id, t)
                    all_histories[hist][t] = feature_vec

                    if t == tags[word_idx]:
                        for entry in feature_vec:
                            feature_counter[entry] += 1

                sample['feature_vectors'][str(word_idx) + '_' + words[word_idx]] = all_histories[hist]

            data[i] = sample

        return data, feature_counter


"""
convert data file into data file , history will be encoded as string
"""


def data_to_saveable_obj(data):
    data_to_save = {}

    for sample_id, sample in data.items():
        for word, hist in sample['histories'].items():
            hist_key = (hist.t_minus_1 + '_' +
                        hist.t_minus_2 + '_' +
                        hist.sentence + '_' +
                        str(hist.index) + '_' +
                        hist.cur_w)

            sample['histories'][word] = hist_key
        data_to_save[sample_id] = sample

    return data_to_save


"""
read json file and convert it into data file 
"""


def from_js_to_data(json_obj):
    data_to_load = {}

    for sample_id, sample in json_obj.items():
        for word, hist_concat in sample['histories'].items():
            hist_key = hist_concat.split('_')
            hist = History(hist_key[0], hist_key[1], hist_key[2], int(hist_key[3]))
            sample['histories'][word] = hist

        data_to_load[sample_id] = sample

    return data_to_load


"""
pre process training data
"""
def preprocess_training_data(dict_percentage, cut_num=100, mode='train1'):
    global word_shape_dict
    global threshold_dict_percentage
    global path

    path = utils.data_path + '/' + mode + '.wtag'
    threshold_dict_percentage = dict_percentage
    word_shape_dict = utils.create_word_shape_dicts()

    if cut_num != -1:
        path = utils.cut_train_file(n=cut_num, path=path)

    stats = FeatureStatistics()
    stats.get_features_from_data(path)

    feature_class = Feature2ID(stats)
    feature_class.create_feature_id_dict()
    feature_to_id = feature_class.feature_to_id
    id_to_feature = feature_class.id_to_feature

    data, feature_counter = create_data_after_preprocess(stats, feature_to_id)
    data_to_save = data_to_saveable_obj(data)

    with open(json_path + '/preprocessed_data_' + mode + '.json', 'w') as f:
        json.dump(data_to_save, f)

    with open(json_path + '/number_of_features_' + mode + '.json', 'w') as f:
        json.dump(len(id_to_feature), f)

    with open(json_path + '/feature_counter_' + mode + '.json', 'w') as f:
        json.dump(feature_counter, f)

    with open(json_path + '/id_to_feature_' + mode + '.json', 'w') as f:
        json.dump(id_to_feature, f)

    with open(json_path + '/feature_to_id_' + mode + '.json', 'w') as f:
        json.dump(feature_to_id, f)

    return {'preprocessed_data': data_to_save,
            'number_of_features': len(id_to_feature),
            'feature_counter': feature_counter}


"""
pre process evaluation data
"""


def preprocess_eval_data(all_tags, feature_to_id, eval_path):
    data = {}
    all_histories = defaultdict(dict)
    global word_shape_dict
    word_shape_dict = utils.create_word_shape_dicts()

    with open(eval_path) as f:
        i = -1
        for line in f:
            i += 1

            if "\n" == line[-1]:
                line = line[:-1]

            splited_words = line.split(' ')

            sentence = ''
            tags = []
            words = []

            for word_idx in range(len(splited_words)):
                cur_word, cur_tag = splited_words[word_idx].split('_')
                sentence += ' ' + cur_word
                tags.append(cur_tag)
                words.append(cur_word)

            sentence = sentence[1:]
            sample = {'id': i, 'sentence': sentence, 'tags': tags, 'feature_vectors': {}}

            for word_idx in range(len(splited_words)):
                if word_idx == 0:
                    t_minus_1 = '*'
                    t_minus_2 = '*'

                elif word_idx == 1:
                    t_minus_1 = tags[word_idx - 1]
                    t_minus_2 = '*'

                else:
                    t_minus_1 = tags[word_idx - 1]
                    t_minus_2 = tags[word_idx - 2]

                hist = History(t_minus_1, t_minus_2, sentence, word_idx)

                all_histories[hist] = {}

                for t in all_tags:
                    feature_vec = hist.create_feature_vec(feature_to_id, t)
                    all_histories[hist][t] = feature_vec

                sample['feature_vectors'][str(word_idx) + '_' + words[word_idx]] = all_histories[hist]

            data[i] = sample

    return data


"""
pre process compatition data
"""


def preprocess_comp_data(eval_path):
    data = {}
    global word_shape_dict
    word_shape_dict = utils.create_word_shape_dicts()

    with open(eval_path) as f:
        i = -1
        for line in f:
            i += 1
            if "\n" == line[-1]:
                line = line[:-1]

            splited_words = line.split(' ')

            sentence = ''
            words = []

            for w in splited_words:
                sentence += ' ' + w
                words.append(w)

            sentence = sentence[1:]
            sample = {'id': i, 'sentence': sentence}

            data[i] = sample
    return data
