json_main_path = 'Jsons'
trained_model_main_path = 'models'
data_path = 'Data'

"""
model 1 threshold dict
"""
best_model_1_percentages = \
    {'tags_vocab': 1,
     'words_tags': 0.75,
     'bigram_tags': 0.75,
     'trigram_tags': 0.75,
     'ti_suffix_len_4': 0.65,
     'ti_prefix_len_4': 0.65,
     'ti_suffix_len_3': 0.35,
     'ti_prefix_len_3': 0.35,
     'ti_suffix_len_2': 0.15,
     'ti_prefix_len_2': 0.15,
     'ti_suffix_len_1': 0.1,
     'ti_prefix_len_1': 0.1,
     'ti_capital': 1,
     'ti_num': 1,
     'ti_wi_full_shape': 0.0,
     'ti_wi_small_shape': 0.0,
     'ti_wi_minus1': 0.75,
     'ti_wi_plus1': 0.75,
     'ti_first_cap': 1,
     'ti_first_w': 1,
     'ti_last_w': 1,
     'ti_all_cap': 1,
     'ti_all_small': 1,
     'ti_has_hyphen': 1,
     'ti_wi_minus2_wi_minus_1_wi': 0,
     'ti_wi_minus2_wi_minus_1': 0,
     'ti_wi_minus_1_wi': 0
     }

"""
model 2 threshold dict
"""
best_model_2_percentages = \
    {'tags_vocab': 1,
     'words_tags': 1,
     'bigram_tags': 1,
     'trigram_tags': 1,
     'ti_suffix_len_4': 1,
     'ti_prefix_len_4': 1,
     'ti_suffix_len_3': 1,
     'ti_prefix_len_3': 1,
     'ti_suffix_len_2': 1,
     'ti_prefix_len_2': 1,
     'ti_suffix_len_1': 1,
     'ti_prefix_len_1': 1,
     'ti_capital': 1,
     'ti_num': 1,
     'ti_wi_full_shape': 1,
     'ti_wi_small_shape': 1,
     'ti_wi_minus1': 1,
     'ti_wi_plus1': 1,
     'ti_first_cap': 1,
     'ti_first_w': 1,
     'ti_last_w': 1,
     'ti_all_cap': 1,
     'ti_all_small': 1,
     'ti_has_hyphen': 1,
     'ti_wi_minus2_wi_minus_1_wi': 0.25,
     'ti_wi_minus2_wi_minus_1': 0.25,
     'ti_wi_minus_1_wi': 0.25
     }

"""
cut train file for debug
"""


def cut_train_file(n=100, path=''):
    lines = []
    counter = 0
    output_file_name = 'cut_train_' + str(n) + '.wtag'
    output = open(output_file_name, 'w')

    with open(path) as f:
        for line in f:
            lines.append(line)
            counter += 1
            if counter > n:
                break

    output.writelines(lines)
    path = output_file_name
    output.close()
    return path

"""
get top k elements from dict
k - threshold percentage
"""


def get_top_k_percentage(d, percentage, name=''):
    k = int(percentage * len(d))
    if name:
        print('\n\n name: {}\nk: {}\n len: {}'.format(name, k, len(d)))
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:k])


"""
get top k elements from dict
k - threshold number
"""


def get_top_k(d, k, name=''):
    if name:
        print('\n\n name: {}\nk: {}\n len: {}'.format(name, k, len(d)))
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:k])


"""
get top k elements from dict
k - threshold number
"""


def get_top_k_threshold(d, threshold, name=''):
    if name:
        print('\n\n name: {}\n len: {}'.format(name, len(d)))
    new_d = {}
    for key, val in d.items():
        if val > threshold:
            new_d[key] = val
    return new_d


"""
check if word contains number
"""


def contain_num(w):
    for num in range(10):
        if str(num) in w:
            return True
    return False


"""
check if word contains capital letter
"""


def contain_capital(w):
    if w == w.lower():
        return False
    return True


"""
create the shape mapping dict:
capital letter - X
small letter - x
number - d
"""


def create_word_shape_dicts():
    d = {}
    lower = [s_l for s_l in 'abcdefghijklmnopqrstuvwxyz']
    upper = [s_l.upper() for s_l in lower]
    num = [n for n in '0123456789']

    for s_l in lower:
        d[s_l] = 'x'
    for L in upper:
        d[L] = 'X'
    for n in num:
        d[n] = 'd'

    return d


"""
create full shape of a word - XXXdX etc.
"""


def get_word_full_shape(w, word_shape_dict):
    new_w = ''

    for letter in w:
        if letter in word_shape_dict:
            new_l = word_shape_dict[letter]
        else:
            new_l = letter

        new_w = new_w + new_l
    return new_w


"""
create small shape of a word - XdX etc.
"""


def get_word_small_shape(full_shape):
    if full_shape == '':
        return ''

    new_w = full_shape[0]

    for letter in full_shape:
        if letter != new_w[-1]:
            new_w = new_w + letter

    return new_w


"""
get full shape and small shape of a word
"""


def get_word_shape(w, word_shape_dict):
    full_shape = get_word_full_shape(w, word_shape_dict)
    small_shape = get_word_small_shape(full_shape)
    return full_shape, small_shape


"""
check if all letters are capital letters
"""


def check_all_cap_all_small(cur_word):
    all_cap = True
    all_small = True
    for ltr in cur_word:
        if ltr == ltr.lower():
            all_cap = False
        else:
            all_small = False

    return all_cap, all_small
