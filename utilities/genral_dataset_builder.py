import numpy as np
import pickle
import os
from tqdm import tqdm
from collections import Counter
from collections import defaultdict

from pytorch_pretrained_bert import BertTokenizer


class SentFeatures():
    """A sigle set of feature of a sentence"""

    def __init__(self, input_ids, input_mask, segment_ids,
                 text=None, sent_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.text = text
        self.sent_id = sent_id


def sent_processor(sent, tokenizer, max_seq_len):
    tokens = tokenizer.tokenize(sent)
    if len(tokens) > max_seq_len-2:
        tokens = tokens[:max_seq_len-2]
    sent_feature = add_notation(tokens, tokenizer, max_seq_len)
    sent_feature.text = sent

    return sent_feature


def add_notation(tokens, tokenizer, max_seq_len):
    tokens.insert(0, "[CLS]")
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # zero-pad
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
    segment_ids = [0] * max_seq_len

    return SentFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids)


def load_data(max_seq_len, all_sent):
    sequence_len = 50
    splited_file_name = 'Dataset/digit_dataset.pkl'
    saved = False

    if os.path.isfile(splited_file_name) and saved:
        with open(splited_file_name, 'rb') as f:
            train_dataset, valid_dataset, test_dataset = pickle.load(f)
    else:
        print(os.getcwd())
        user_mentioned_dict = get_mentioned_dict()

        print("collect user history")
        user_text_history_dict = get_user_history_dict(max_seq_len)

        print("collect ment history")
        mentioned_text_history_dict = get_mentioned_history_dict(max_seq_len)

        dataset, maxlen, maxmentioned \
            = load_data_and_labels(sequence_len, user_mentioned_dict,
                                   user_text_history_dict,
                                   mentioned_text_history_dict,
                                   max_seq_len, all_sent)
        # dataset: [(), (), (), ...]
        # (): (num, [sent_obj], sent_obj, num, dict_of_list, num)

        userlist = list(set([data[0] for data in dataset]))
        # use set to make list member be unique
        print('Data size: %d, Max tweet length: %d' % (len(dataset), maxlen))
        print('Max mentioned count: %d' % maxmentioned)

        maxlen = 33
        # truncated

        np.random.seed(12345)
        np.random.shuffle(userlist)
        np.random.shuffle(dataset)
        split_index = int(len(userlist)*0.8)
        train_user_id = set(userlist[:split_index])
        test_uset_id = set(userlist[split_index:])

        temp_data = [data for data in dataset if data[0] in train_user_id]
        np.random.shuffle(temp_data)
        split_index = int(len(temp_data)*0.8)
        train_data = temp_data[:split_index]
        valid_data = temp_data[split_index:]
        test_data = defaultdict(list)
        for data in dataset:
            if data[0] in test_uset_id:
                test_data[data[0]].append(data)

        print("Build training data")
        train_dataset = build_train_data(train_data)
        # train_data:[[], [], [], ...]
        # []: [num, num, sent_obj, [sent_obj], [sent_obj], num]

        print("Build valid data")
        valid_dataset = build_valid_data(valid_data)
        # valid_data: [*, *, *, ...]
        # *: train_data

        print("Build test data")
        test_dataset = build_test_data(test_data)
        # test_data: [*, *, *, ...]
        # *: [train_data, valid_data]

        return dataset, train_dataset, valid_dataset, test_dataset


def get_mentioned_dict():
    user_mentioned_dict = defaultdict(list)
    mentioned_reader = open('../Dataset/user_mentioned.txt', 'r')

    for line in mentioned_reader:
        line = line.replace('\n', ' ').strip().split('\t')
        uid = line[0]
        mentioned = line[1:]
        user_mentioned_dict[uid] = mentioned
    mentioned_reader.close()
    return user_mentioned_dict


def get_user_history_dict(max_seq_len):
    user_text_history_dict = defaultdict(list)
    history_reader = open(
        '../Dataset/author/users_history_random_201.txt', 'r')
    tokenizer = BertTokenizer.from_pretrained(
        "/home/pl/bert/uncased_L-12_H-768_A-12")
    for line in tqdm(history_reader):
        line = line.strip().split('\t')
        uid = line[0]
        history_tweet = line[1].lower().strip()
        history_tweet = sent_processor(
            history_tweet, tokenizer, max_seq_len)
        user_text_history_dict[uid].append(history_tweet)
    history_reader.close()
    return user_text_history_dict


def get_mentioned_history_dict(max_seq_len):
    mentioned_text_history_dict = defaultdict(list)
    history_reader = open(
        '../Dataset/mentioned_user/mentioned_history_random_200.txt', 'r')
    tokenizer = BertTokenizer.from_pretrained(
        "/home/pl/bert/uncased_L-12_H-768_A-12")
    for line in tqdm(history_reader):
        line = line.strip().split('\t')
        uid = line[0]
        history_tweet = line[1].lower().strip()
        history_tweet = sent_processor(
            history_tweet, tokenizer, max_seq_len)
        mentioned_text_history_dict[uid].append(history_tweet)
    history_reader.close()
    return mentioned_text_history_dict


# load_data_and_labels() is use to generate raw data
def load_data_and_labels(sequence_len, user_mentioned_dict,
                         user_text_history_dict, mentioned_text_history_dict,
                         max_seq_len, all_sent):
    tweet_reader = open('../Dataset/tweet/tweet_data.txt', 'r')
    dataset = []
    maxlen = 33
    maxmentioned = 0
    tokenizer = BertTokenizer.from_pretrained(
        "/home/pl/bert/uncased_L-12_H-768_A-12")
    for line in tqdm(tweet_reader):
        line = line.replace('\n', ' ').strip().split('\t')
        uid = line[1]
        if uid not in user_mentioned_dict.keys():
            continue
        tweet_x = line[2].lower().strip()

        tweet_x = sent_processor(
            tweet_x, tokenizer, max_seq_len)
        collect_unique_sent(tweet_x, all_sent)

        um = user_mentioned_dict[uid]

        if maxmentioned < len(um):
            maxmentioned = len(um)

        y = line[3].split('||')

        user_x = user_text_history_dict[uid][:sequence_len]
        for sent in user_x:
            collect_unique_sent(sent, all_sent)

        mentioned_text_history = defaultdict(list)

        for mentioned_id in um:
            mentioned_text_history[mentioned_id] \
                = mentioned_text_history_dict[mentioned_id][:sequence_len]
            for sent in mentioned_text_history[mentioned_id]:
                collect_unique_sent(sent, all_sent)

        dataset.append((uid, user_x, tweet_x, um, mentioned_text_history, y))

    return [dataset, maxlen, maxmentioned]


def collect_unique_sent(sent_feature, all_sent):
    if sent_feature not in all_sent:
        all_sent.add(sent_feature)


def build_train_data(dataset):
    uid_x = []
    tweet_x = []
    user_x = []
    mentioned_id = []
    mentioned_x = []
    y = []
    train_data = []
    for data in tqdm(dataset):
        uid, ux, tx, um, mx, target = data
        for m_id in um:
            example = list()
            example.append(uid)
            example.append(m_id)
            example.append(tx)
            example.append(ux)
            example.append(mx[m_id])
            if m_id in target:
                example.append(1)
            else:
                example.append(0)
            train_data.append(example)

    return train_data


def build_valid_data(dataset):
    valid_data = []
    for data in tqdm(dataset):
        user = list()
        uid, ux, tx, um, mx, target = data
        for m_id in um:
            example = list()
            example.append(uid)
            example.append(m_id)
            example.append(tx)
            example.append(ux)
            example.append(mx[m_id])
            if m_id in target:
                example.append(1)
            else:
                example.append(0)
            user.append(example)

        valid_data.append(user)

    return valid_data


def build_test_data(dataset):
    test_data = []
    for uid, userdata in tqdm(dataset.items()):
        split_index = int(len(userdata)*0.8)
        train_data = userdata[:split_index]
        valid_data = userdata[split_index:]

        train_dataset = build_train_data(train_data)
        valid_dataset = build_valid_data(valid_data)
        test_data.append([train_dataset, valid_dataset])
    return test_data


def modify_data():
    print(os.getcwd())

    with open("../Dataset/digit_data.pkl", 'rb') as f:
        dataset = pickle.load(f)
    # dataset: [(), (), (), ...]
    # (): (num, [sent_obj], sent_obj, num, dict_of_list, num)

    dataset = build_train_data(dataset)
    user_list = list(set([data[0] for data in dataset]))
    all_users = list(set([data[1] for data in dataset]))
    all_users.extend(user_list)
    all_users = list(set(all_users))
    print(f"Total tweet users {len(user_list)}, total used users {len(all_users)}")

    np.random.seed(12345)
    np.random.shuffle(dataset)
    split_8_index = int(len(dataset)*0.8)
    split_9_index = int(len(dataset)*0.9)

    train_data = dataset[:split_8_index]
    valid_data = dataset[split_8_index:split_9_index]
    test_data = dataset[split_9_index:]
    print(f"n/p in train: {calu_data_balence(train_data)}, dev: {calu_data_balence(valid_data)}, test: {calu_data_balence(test_data)}")

    # train_data:[[], [], [], ...]
    # []: [num, num, sent_obj, [sent_obj], [sent_obj], num]

    return train_data, valid_data, test_data


def calu_data_balence(dataset):
    p, n = 0, 0
    for d in dataset:
        if d[-1] == 1:
            p += 1
        else:
            n += 1
    return n/p


if __name__ == '__main__':

    train_dataset, valid_dataset, test_dataset = modify_data()
    with open("../Dataset/digit_data_train.pkl", 'wb') as f:
        pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("../Dataset/digit_data_valid.pkl", 'wb') as f:
        pickle.dump(valid_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("../Dataset/digit_data_test.pkl", 'wb') as f:
        pickle.dump(test_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
