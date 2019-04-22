import time
import imp
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import json
import tqdm

from utilities.utils import *


result_log = "Top_{}: precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, " + \
    "hits_score: {:.4f}, hits@3 score: {:.4f}, hits@5 score: {:.4f}"


def train_marl(
    embedding, model, train_data, selected_data, valid_data, test_data,
        l2, device, model_name):
    best_f1 = 0
    best_test_f1 = 0

    logging.info(f"Joint training, now is {time.ctime(time.time())}")
    weight = 0
    rl_epoch = 0
    try:
        for epoch in count(1):
            logging.info(f"Epoch: {epoch}")
            if epoch <= 3:
                pretrain_epoch(embedding, model, train_data, device)
                benchmark_epoch(embedding, model, valid_data, device)
                benchmark_epoch(embedding, model, test_data, device)
            else:
                rl_only_train(embedding, model, train_data, device, weight)
            if epoch % 1 == 0:
                f1 = valid_epoch(embedding, model, valid_data, device)
                test_f1 = valid_epoch(embedding, model, test_data, device)
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model, f"parameter/{model_name}")
                    logging.info("Weights saved @ f1 = {:.4f}".format(best_f1))
                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                    torch.save(model, f"parameter/{model_name}_test")
                    logging.info("test f1 {:.4f}".format(best_test_f1))
    except KeyboardInterrupt as e:
        return best_f1


def joint_train_epoch(embedding, model, data_loader, device, verbose=1):
    loss_total, acc_num, total_num = [], [], []
    c_loss_total, a_loss_total, m_loss_total = [], [], []
    total_num = 0
    tp = 0
    precision_denominator = 0
    recall_denominator = 0
    precision = 0
    recall = 0
    f1 = 0
    user_selected_num = 0
    ment_selected_num = 0

    for batch_idx, data in enumerate(data_loader):
        tweet, user_h, ment_h, target = data
        with torch.no_grad():
            tweet = embedding(tweet)
            user_h = embedding(user_h)
            ment_h = embedding(ment_h)
        data = (tweet, user_h, ment_h, target)
        cf_loss, _, batch_acc, target, \
            selected_num, predict = model.update_buffer(
                data, device, need_backward=True,
                train_classifier=True, update_buffer=True)
        user_selected_num += selected_num[0]
        ment_selected_num += selected_num[1]

        batch_acc = batch_acc.float()
        cr_loss, u_loss, m_loss, u_A, m_A = model.optimize_AC(device)

        acc_num.append(
            torch.sum(batch_acc.long()).item())
        total_num += len(batch_acc)

        loss_total.append(cf_loss)
        c_loss_total.append(cr_loss)
        a_loss_total.append(u_loss)
        m_loss_total.append(m_loss)

        tp += torch.sum(batch_acc * target).item()
        precision_denominator += torch.sum(predict).item()
        recall_denominator += torch.sum(target).item()
        if precision_denominator > 0:
            precision = tp / precision_denominator
        if recall_denominator > 0:
            recall = tp / recall_denominator
        if precision > 0 or recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        if verbose > 0:
            message = "Joint_Train: [{}/{}, {:.2f}%], ".format(
                    batch_idx+1, len(data_loader),
                    100 * (batch_idx+1) / len(data_loader)
                ) + \
                "cf_l: {:.4f}, c_l: {:.2e}, u_l: {:.2e}, m_l: {:.2e}, acc: {:.4f}, ".format(
                    sum(loss_total)/(batch_idx+1),
                    sum(c_loss_total)/(batch_idx+1),
                    sum(a_loss_total)/(batch_idx+1),
                    sum(m_loss_total)/(batch_idx+1),
                    sum(acc_num)/total_num
                ) + \
                "p: {:.4f}, r: {:.4f}, f1: {:.4f}, ".format(
                    precision, recall, f1
                ) + \
                "u_s: {:.2f}, m_s: {:.2f}    ".format(
                    user_selected_num/(batch_idx+1),
                    ment_selected_num/(batch_idx+1)
                )
            print(message, end='\r')
    if verbose > 0:
        logging.info(message)


def valid_epoch(embedding, model, data_loader, device):
    y_pred, y_test, acc_num = [], [], []
    total_num = 0
    tp = 0
    precision_denominator = 0
    recall_denominator = 0
    precision = 0
    recall = 0
    f1 = 0
    user_selected_num = 0
    ment_selected_num = 0

    for batch_idx, data in enumerate(data_loader):
        tweet, user_h, ment_h, target = data
        with torch.no_grad():
            tweet = embedding(tweet)
            user_h = embedding(user_h)
            ment_h = embedding(ment_h)
            user_h, ment_h = [
                torch.chunk(item, 10, dim=1)[0]
                for item in [user_h, ment_h]]
            data = (tweet, user_h, ment_h, target)
            _, output, batch_acc, target, \
                selected_num, predict = model.update_buffer(
                    data, device, need_backward=False,
                    train_classifier=False, update_buffer=False)

            acc_num.append(
                torch.sum(batch_acc.long()).item())
            total_num += len(batch_acc)
            user_selected_num += selected_num[0]
            ment_selected_num += selected_num[1]

            batch_acc = batch_acc.float()

            tp += torch.sum(batch_acc * target).item()
            precision_denominator += torch.sum(predict).item()
            recall_denominator += torch.sum(target).item()
            if precision_denominator > 0:
                precision = tp / precision_denominator
            if recall_denominator > 0:
                recall = tp / recall_denominator
            if precision > 0 or recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)

            message = "Test: [{}/{}, {:.2f}%], ".format(
                    batch_idx+1, len(data_loader),
                    100 * (batch_idx+1) / len(data_loader)
                ) + \
                "acc: {:.4f}, ".format(
                    sum(acc_num)/total_num
                ) + \
                "p: {:.4f}, r: {:.4f}, f1: {:.4f}, ".format(
                    precision, recall, f1
                ) + \
                "u_s: {:.2f}, m_s: {:.2f}".format(
                    user_selected_num/(batch_idx+1),
                    ment_selected_num/(batch_idx+1)
                )
            print(message, end='\r')
    logging.info(message)

    return f1


def rl_only_train(embedding, model, data_loader, device, weight=0):
    loss_total, acc_num, total_num = [], [], []
    u_A_total, m_A_total = [], []
    c_loss_total, a_loss_total, m_loss_total = [], [], []
    total_num = 0
    tp = 0
    precision_denominator = 0
    recall_denominator = 0
    precision = 0
    recall = 0
    f1 = 0
    user_selected_num = 0
    ment_selected_num = 0

    for batch_idx, data in enumerate(data_loader):
        tweet, user_h, ment_h, target = data
        with torch.no_grad():
            tweet = embedding(tweet)
            user_h = embedding(user_h)
            ment_h = embedding(ment_h)
            user_h, ment_h = [
                torch.chunk(item, 10, dim=1)[0]
                for item in [user_h, ment_h]]
            data = (tweet, user_h, ment_h, target)

            cf_loss, _, batch_acc, target, \
                selected_num, predict = model.update_buffer(
                    data, device, need_backward=True,
                    train_classifier=False, update_buffer=True)
        user_selected_num += selected_num[0]
        ment_selected_num += selected_num[1]

        batch_acc = batch_acc.float()

        cr_loss, u_loss, m_loss, u_A, m_A = model.optimize_AC(device, weight)

        acc_num.append(
            torch.sum(batch_acc.long()).item())
        total_num += len(batch_acc)

        loss_total.append(cf_loss)
        c_loss_total.append(cr_loss)
        a_loss_total.append(u_loss)
        m_loss_total.append(m_loss)
        u_A_total.append(u_A)
        m_A_total.append(m_A)

        tp += torch.sum(batch_acc * target).item()
        precision_denominator += torch.sum(predict).item()
        recall_denominator += torch.sum(target).item()
        if precision_denominator > 0:
            precision = tp / precision_denominator
        if recall_denominator > 0:
            recall = tp / recall_denominator
        if precision > 0 or recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        message = "RL_Train: [{}/{}, {:.2f}%], ".format(
                batch_idx+1, len(data_loader),
                100 * (batch_idx+1) / len(data_loader)
            ) + \
            "cf_l: {:.4f}, c_l: {:.2e}, u_l: {:.2e}, m_l: {:.2e} acc: {:.4f}, ".format(
                sum(loss_total)/(batch_idx+1),
                sum(c_loss_total)/(batch_idx+1),
                sum(a_loss_total)/(batch_idx+1),
                sum(m_loss_total)/(batch_idx+1),
                sum(acc_num)/total_num
            ) + \
            "p: {:.4f}, r: {:.4f}, f1: {:.4f}, ".format(
                precision, recall, f1
            ) + \
            "u_s: {:.2f}, m_s: {:.2f}, ".format(
                user_selected_num/(batch_idx+1),
                ment_selected_num/(batch_idx+1)
            ) + \
            "u_A: {:.2e}, m_A: {:.2e}".format(
                sum(u_A_total)/(batch_idx+1),
                sum(m_A_total)/(batch_idx+1)
            )
        print(message, end='\r')
    logging.info(message)


def benchmark_epoch(embedding, model, data_loader, device):
    y_pred, y_test, acc_num = [], [], []
    total_num = 0
    tp = 0
    precision_denominator = 0
    recall_denominator = 0
    precision = 0
    recall = 0
    f1 = 0

    for batch_idx, data in enumerate(data_loader):
        tweet, user_h, ment_h, target = data
        with torch.no_grad():
            tweet = embedding(tweet)
            user_h = embedding(user_h)
            ment_h = embedding(ment_h)
            user_h, ment_h = [
                torch.chunk(item, 10, dim=1)[0]
                for item in [user_h, ment_h]]
            data = (tweet, user_h, ment_h, target)
            _, output, batch_acc, target, predict = model.classifier_benchmark(
                    data, device)

            batch_acc = batch_acc.float()

            acc_num.append(
                torch.sum(batch_acc.long()).item())
            total_num += len(batch_acc)

            tp += torch.sum(batch_acc * target).item()
            precision_denominator += torch.sum(predict).item()
            recall_denominator += torch.sum(target).item()
            if precision_denominator > 0:
                precision = tp / precision_denominator
            if recall_denominator > 0:
                recall = tp / recall_denominator
            if precision > 0 or recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)

            message = "Benchmark: [{}/{}, {:.2f}%], ".format(
                    batch_idx+1, len(data_loader),
                    100 * (batch_idx+1) / len(data_loader)
                ) + \
                "acc: {:.4f}, ".format(
                    sum(acc_num)/total_num
                ) + \
                "p: {:.4f}, r: {:.4f}, f1: {:.4f}, ".format(
                    precision, recall, f1
                )
            print(message, end='\r')
    logging.info(message)


def classifier_train_epoch(
    embedding, model, data_loader, device, verbose=1, percent=1
):
    loss_total, acc_num, total_num = [], [], []
    c_loss_total, a_loss_total = [], []
    total_num = 0
    tp = 0
    precision_denominator = 0
    recall_denominator = 0
    precision = 0
    recall = 0
    f1 = 0
    user_selected_num = 0
    ment_selected_num = 0
    total_batch = len(data_loader) * percent

    for batch_idx, data in enumerate(data_loader):
        if batch_idx > total_batch:
            break
        tweet, user_h, ment_h, target = data
        with torch.no_grad():
            tweet = embedding(tweet)
            user_h = embedding(user_h)
            ment_h = embedding(ment_h)
        data = (tweet, user_h, ment_h, target)
        cf_loss, _, batch_acc, target, \
            selected_num, predict = model.update_buffer(
                data, device, need_backward=True,
                train_classifier=True, update_buffer=False)
        user_selected_num += selected_num[0]
        ment_selected_num += selected_num[1]

        batch_acc = batch_acc.float()

        acc_num.append(
            torch.sum(batch_acc.long()).item())
        total_num += len(batch_acc)

        loss_total.append(cf_loss)

        tp += torch.sum(batch_acc * target).item()
        precision_denominator += torch.sum(predict).item()
        recall_denominator += torch.sum(target).item()
        if precision_denominator > 0:
            precision = tp / precision_denominator
        if recall_denominator > 0:
            recall = tp / recall_denominator
        if precision > 0 or recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        if verbose > 0:
            message = "Classifier_Train: [{}/{}, {:.2f}%], ".format(
                    batch_idx+1, len(data_loader),
                    100 * (batch_idx+1) / len(data_loader)
                ) + \
                "cf_l: {:.4f}, acc: {:.4f}, ".format(
                    sum(loss_total)/(batch_idx+1),
                    sum(acc_num)/total_num
                ) + \
                "p: {:.4f}, r: {:.4f}, f1: {:.4f}, ".format(
                    precision, recall, f1
                ) + \
                "u_s: {:.2f}, m_s: {:.2f}".format(
                    user_selected_num/(batch_idx+1),
                    ment_selected_num/(batch_idx+1)
                )
            print(message, end='\r')
    if verbose > 0:
        logging.info(message)


def pretrain_epoch(embedding, model, data_loader, device):
    loss_total, acc_num, total_num = [], [], 0
    tp = 0
    precision_denominator = 0
    recall_denominator = 0
    precision = 0
    recall = 0
    f1 = 0

    for batch_idx, data in enumerate(data_loader):
        tweet, user_h, ment_h, target = data
        with torch.no_grad():
            tweet = embedding(tweet)
            user_h = embedding(user_h)
            ment_h = embedding(ment_h)
            user_h, ment_h = [
                torch.chunk(item, 10, dim=1)[0]
                for item in [user_h, ment_h]]
            data = (tweet, user_h, ment_h, target)

        cf_loss, output, batch_acc, target, predict = model.pre_train_classifier(
                data, device)

        batch_acc = batch_acc.float()
        loss_total.append(cf_loss)
        acc_num.append(
            torch.sum(batch_acc.long()).item())
        total_num += len(batch_acc)

        tp += torch.sum(batch_acc * target).item()
        precision_denominator += torch.sum(predict).item()
        recall_denominator += torch.sum(target).item()
        if precision_denominator > 0:
            precision = tp / precision_denominator
        if recall_denominator > 0:
            recall = tp / recall_denominator
        if precision > 0 or recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        message = "Pretrain: [{}/{}, {:.2f}%], ".format(
                batch_idx+1, len(data_loader),
                100 * (batch_idx+1) / len(data_loader)
            ) + \
            "cf_l: {:.4f}, acc: {:.4f}, ".format(
                sum(loss_total)/(batch_idx+1),
                sum(acc_num)/total_num
            ) + \
            "p: {:.4f}, r: {:.4f}, f1: {:.4f}, ".format(
                precision, recall, f1
            )
        print(message, end='\r')
    logging.info(message)


def create_step_schedule(initial, step):
    state = [initial, 1e9, step, 0]

    def step_schedule(observation):
        if observation < state[1]:
            state[1] = observation
            state[3] = 0
        else:
            state[3] += 1
        if state[3] >= state[2]:
            state[0] *= 0.1
            state[3] = 0

        return state[0]

    return step_schedule


def update_config(old_config, model):
    imp.reload(utilities.train_config)
    new_config = utilities.train_config.fast_config()
    if old_config.rl_lr != new_config.rl_lr:
        model.modify_rl_lr(new_config.rl_lr)
    if old_config.classifier_lr != new_config.classifier_lr:
        model.modify_classifier_lr(new_config.classifier_lr)

    return new_config


def collect_indicator(embedding, model, data_loader, device):
    indicators = list()
    indicator_count = 0

    for batch_idx, data in enumerate(data_loader):
        uid, mid, tweet, user_h, ment_h, target = data
        sent_id = tweet.item()
        with torch.no_grad():
            tweet = embedding(tweet)
            user_h = embedding(user_h)
            ment_h = embedding(ment_h)
            data = (tweet, user_h, ment_h, target)

            indicator, flag = model.find_indicator(data, device)

            if flag is True:
                indicator_count += 1
                indicator["uid"] = uid
                indicator["mid"] = mid
                indicator["sent_id"] = sent_id
                logging.info(f"find {indicator_count}_th indicator")
                indicators.append(indicator)
        print(f"{batch_idx}, {len(data_loader)}")

    with open("indicator.json", "w") as f:
        json.dump(indicators, f)
        logging.info("indicators saved")
