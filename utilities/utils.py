import torch
import numpy as np


def precision_score(y_test, y_pred, k=1):
    p_score = []
    for i in range(len(y_test)):
        if len(y_test[i]) < k:
            continue
        _, result_at_topk = y_pred[i].topk(k)
        count = 0
        for j in result_at_topk:
            if y_test[i][j] == 1:
                count += 1
        p_score.append(count / k)
    return np.mean(p_score)


def recall_score(y_test, y_pred, k=1):
    r_score = []
    for i in range(len(y_test)):
        if len(y_test[i]) < k:
            continue
        _, result_at_topk = y_pred[i].topk(k)
        count = 0
        for j in result_at_topk:
            if y_test[i][j] == 1:
                count += 1
        r_score.append(count / y_test[i].sum())

    return np.mean(r_score)


def hits_score(y_test, y_pred, k=1):
    h_score = []
    for i in range(len(y_test)):
        if len(y_test[i]) < k:
            continue
        _, result_at_topk = y_pred[i].topk(k)
        count = 0
        for j in result_at_topk:
            if y_test[i][j] == 1:
                count += 1
        h_score.append(1 if count > 0 else 0)

    return np.mean(h_score)


def mrr_score(y_test, y_pred):
    m_score = []
    for i in range(len(y_test)):
        _, result_at_topk = y_pred[i].topk(len(y_pred[i]))
        for j in range(len(y_pred[i])):
            if y_test[i][result_at_topk[j]] == 1:
                m_score.append(1.0 / (j+1))
                break

    return np.mean(m_score)


def bpref(y_test, y_pred):
    b_score = []
    for i in range(len(y_test)):
        index = 0
        _, result_at_topk = y_pred[i].topk(len(y_pred[i]))
        for j in range(len(y_pred[i])):
            if y_test[i][result_at_topk[j]] == 1:
                index = j+1
        b_score.append(1.0 - (index - y_test[i].sum())/len(y_pred[i]))

    return np.mean(b_score)


def dhms_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    return f"{d} day {h} h {m} m {s} s"


def precision(y_test, y_pred):
    pass


def recall(y_test, y_pred):
    pass
