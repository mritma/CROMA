import pickle
import random
import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


r"""
    train_dataset: uid_x, tweet_x, user_x, mentioned_id, mentioned_x, y
        all are numpy.ndarray with dtype=int32
        uid_x: (1063271,)
        tweet_x: (1063271, 31)
        user_x: (1063271, 155)
        mentioned_id: (1063271,)
        mentioned_x: (1063271, 155)
        y: (1063271,)
        train set: prediction with one user_mention pair
    valid_dataset:
        valid_dataset[i]: a tweet with multi candidate
        valid_dataset[i] is a tuple, and has the same structure with
            train_dataset
        valid_dataset is a list of tuples
    """


def load_data(batch_size, debug=False, test=False):
    if test:
        with open("Dataset/old_test_data.pkl", 'rb') as f:
            old_test_dataset = pickle.load(f)
        with open("Dataset/digit_data_test.pkl", 'rb') as f:
            test_dataset = pickle.load(f)

        test_data = DataLoader(
            trainDataset(test_dataset),
            batch_size=batch_size, shuffle=False,
            collate_fn=train_collate,
            num_workers=min(batch_size, 4)
        )
        old_test_data = DataLoader(
            validDataset(old_test_dataset),
            batch_size=1, shuffle=False,
            collate_fn=valid_collate
        )

        return test_data, old_test_data

    with open("Dataset/digit_data_train.pkl", 'rb') as f:
        train_dataset = pickle.load(f)
    with open("Dataset/digit_data_valid.pkl", 'rb') as f:
        valid_dataset = pickle.load(f)
    with open("Dataset/digit_data_test.pkl", 'rb') as f:
        test_dataset = pickle.load(f)

    
    select_dataset = valid_dataset[:batch_size*87]

    if debug:
        train_dataset = train_dataset[:batch_size*15]
        valid_dataset = valid_dataset[:batch_size*15]
        test_dataset = test_dataset[:batch_size*15]
        valid_pos = sum([i[-1] for i in valid_dataset if i[-1] == 1])
        test_pos = sum([i[-1] for i in test_dataset if i[-1] == 1])
        logging.info(f"pos in valid: {valid_pos}, in test: {test_pos}")

    train_data = DataLoader(trainDataset(train_dataset),
                            batch_size=batch_size, shuffle=True,
                            collate_fn=train_collate,
                            num_workers=min(batch_size, 4))

    valid_data = DataLoader(trainDataset(valid_dataset),
                            batch_size=batch_size, shuffle=False,
                            collate_fn=train_collate,
                            num_workers=min(batch_size, 4))

    test_data = DataLoader(trainDataset(test_dataset),
                           batch_size=batch_size, shuffle=False,
                           collate_fn=train_collate,
                           num_workers=min(batch_size, 4))

    select_dataset = DataLoader(trainDataset(select_dataset),
                                batch_size=batch_size, shuffle=False,
                                collate_fn=train_collate,
                                num_workers=min(batch_size, 4))
    return [train_data, valid_data, test_data, select_dataset]


# def load_data(batch_size):
#     with open("Dataset/digit_data_train.pkl", 'rb') as f:
#         train_dataset = pickle.load(f)
#     with open("Dataset/digit_data_valid.pkl", 'rb') as f:
#         valid_dataset = pickle.load(f)
#     with open("Dataset/digit_data_test.pkl", 'rb') as f:
#         test_dataset = pickle.load(f)

#     train_data = DataLoader(trainDataset(train_dataset),
#                             batch_size=batch_size, shuffle=True,
#                             collate_fn=train_collate,
#                             num_workers=min(batch_size, 16))

#     valid_data = DataLoader(validDataset(valid_dataset),
#                             batch_size=1, shuffle=False,
#                             collate_fn=valid_collate)

#     test_data = testDataset(test_dataset, batch_size)

#     return [train_data, valid_data, test_data]


class trainDataset(Dataset):
    def __init__(self, data):
        super(trainDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class validDataset(Dataset):
    def __init__(self, data):
        super(validDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class testDataset(object):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        train_part, test_part = self.data[index]
        train_part = DataLoader(trainDataset(train_part),
                                batch_size=self.batch_size, shuffle=True,
                                collate_fn=train_collate)
        test_part = DataLoader(validDataset(test_part),
                               batch_size=1, shuffle=False,
                               collate_fn=valid_collate)
        return train_part, test_part


def train_collate(batch):
    _, _, t_s, uh_s, mh_s, y = zip(*batch)

    batch_t = torch.LongTensor([t.sent_id for t in t_s])
    batch_uhs = torch.LongTensor(
        [[t.sent_id for t in uh] for uh in uh_s]
    )
    batch_mhs = torch.LongTensor(
        [[t.sent_id for t in mh] for mh in mh_s]
    )

    y = torch.tensor(y, dtype=torch.float)

    return (batch_t, batch_uhs, batch_mhs, y)


def valid_collate(batch):
    assert len(batch) == 1, "The batch size of valid_data must be 1."
    data = batch[0]
    _, _, t_s, uh_s, mh_s, y = zip(*data)

    batch_t = torch.LongTensor([t.sent_id for t in t_s])
    batch_uhs = torch.LongTensor(
        [[t.sent_id for t in uh] for uh in uh_s]
    )
    batch_mhs = torch.LongTensor(
        [[t.sent_id for t in mh] for mh in mh_s]
    )

    y = torch.tensor(y, dtype=torch.float)

    return (batch_t, batch_uhs, batch_mhs, y)


def indicator_collate(batch):
    uid, mid, t_s, uh_s, mh_s, y = zip(*batch)

    batch_t = torch.LongTensor([t.sent_id for t in t_s])
    batch_uhs = torch.LongTensor(
        [[t.sent_id for t in uh] for uh in uh_s]
    )
    batch_mhs = torch.LongTensor(
        [[t.sent_id for t in mh] for mh in mh_s]
    )

    y = torch.tensor(y, dtype=torch.float)

    return (uid, mid, batch_t, batch_uhs, batch_mhs, y)
