import torch
import torch.optim as optim

from models.marl import MultiAgentClassifier
from utilities.utils import *


result_log = "Top_{}: precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, " + \
    "hits_score: {:.4f}, hits@3 score: {:.4f}, hits@5 score: {:.4f}"


def test(test_dataset, args, device, fine_tune=True):
    y_pred = []
    y_test = []
    i = 0
    for train_part, test_part in test_dataset:
        if len(train_part) < 1 or len(test_part) < 1:
            i += 1
            continue

        model = torch.load("parameter/marl")
        model.to(device)

        if fine_tune:
            for epoch in range(1, 16):
                for batch_idx, data in enumerate(train_part):
                    cf_loss, batch_acc = model.joint_train(data, device)

        acc_num = []
        total_num = 0
        for batch_idx, data in enumerate(test_part):
            data = [torch.chunk(item, item.size(0), dim=0)
                    for item in data]
            output_list = []
            target_list = []
            for item in zip(*data):
                _, output, batch_acc, target = model.update_buffer(
                    item, device, need_backward=False,
                    train_classifier=False, update_buffer=False)

                acc_num.append(
                    torch.sum(batch_acc.long()).item())
                total_num += len(target)

                output_list.append(output)
                target_list.append(target)

            output = torch.cat(output_list, dim=0)
            target = torch.cat(target_list, dim=0)
            py = output.data
            y_pred.append(py)
            y_test.append(target)

    if not fine_tune:
        print("Untuned result: ")
    print("skip {} test sample".format(i))
    mrr = mrr_score(y_test, y_pred)
    bp = bpref(y_test, y_pred)
    print("MRR: {}, Bpref: {}".format(mrr, bp))
    topk = [1, 2, 3, 4, 5]
    for k in topk:
        precision = precision_score(y_test, y_pred, k=k)
        recall = recall_score(y_test, y_pred, k=k)
        hscore = hits_score(y_test, y_pred, k=k)
        hits3 = hits_score(y_test, y_pred, k=3)
        hits5 = hits_score(y_test, y_pred, k=5)
        F1 = 2 * (precision * recall) / (precision + recall)
        print(
            result_log.format(k, precision, recall, F1, hscore, hits3, hits5))
