import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

from utilities.utils import *


result_log = "Top_{}: precision: {:.4f}, recall: {:.4f}, F1: {:.4f}, " + \
    "hits_score: {:.4f}, hits@3 score: {:.4f}, hits@5 score: {:.4f}"


def train_classifier(classifier, author_agent, ment_agent, embeddings,
                     train_data, valid_data, l2, device, max_epoch, best_f1=0):
    print(f"Training classifier, now is {time.ctime(time.time())}")
    optimizer = optim.Adam([{"params": classifier.parameters()},
                            {"params": embeddings.parameters()}],
                           weight_decay=l2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.1, patience=2, threshold_mode='abs')
    for epoch in count(1):
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        print("Epoch: ", epoch, " lr: ", current_lr)
        train_epoch(
            classifier, author_agent, ment_agent, embeddings,
            train_data, optimizer, device)
        if epoch % 1 == 0:
            f1 = valid_epoch(
                classifier, author_agent, ment_agent, embeddings,
                valid_data, device)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(
                    classifier.state_dict(), "parameter/classifier")
                torch.save(
                    embeddings.state_dict(), "parameter/embeddings")
                print("Weights saved @ f1 = {:.3f}".format(best_f1))
            scheduler.step(f1)

        if current_lr < 1e-6 or epoch >= max_epoch:
            print(f"Classifier traning is stopped, lr is {current_lr}")
            classifier.load_state_dict(
                torch.load("parameter/classifier"))
            embeddings.load_state_dict(
                torch.load("parameter/embeddings"))
            embeddings.to(device), classifier.to(device)
            print(
                "Load best model in this training episode, f1: {:.3f}"
                .format(best_f1))
            print(f"Classifier training stop at {time.ctime(time.time())}")
            break

    return best_f1


def train_epoch(classifier, author_agent, ment_agent, embeddings,
                data_loader, optimizer, device, verbose=1):
    classifier.train(), embeddings.train()
    author_agent.eval(), ment_agent.eval()
    loss_total, acc_num, total_num = [], [], []
    total_num = 0

    for batch_idx, data in enumerate(data_loader):
        tweet, user_h, ment_h, target = [item.to(device) for item in data]
        # user_h: (batch, 33*200)

        optimizer.zero_grad()
        tweet, user_h, ment_h = map(embeddings, [tweet, user_h, ment_h])
        # user_h: (batch, 33*200, 300)

        user_h = user_h.view(-1, 200, 33, 300)
        ment_h = ment_h.view(-1, 200, 33, 300)

        author_agent.reset_buffer()
        ment_agent.reset_buffer()

        output = classifier(tweet, user_h, ment_h, author_agent, ment_agent)
        output = output.squeeze(dim=-1)
        batch_acc = output.ge(0).float() == target
        acc_num.append(
            torch.sum(batch_acc.long()).item())
        total_num += len(target)

        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        loss_total.append(loss.item())

        if verbose > 0:
            print("Train: [{}/{}, {:.2f}%], loss: {:.4f}, acc: {:.4f}".format(
                batch_idx+1, len(data_loader),
                100 * (batch_idx+1) / len(data_loader),
                sum(loss_total)/(batch_idx+1), sum(acc_num)/total_num),
                end='\r')
    if verbose > 0:
        print()


def valid_epoch(classifier, author_agent, ment_agent,
                embeddings, data_loader, device):
    classifier.eval(), embeddings.eval()
    author_agent.eval(), embeddings.eval()
    y_pred, y_test, acc_num = [], [], []
    total_num = 0

    for batch_idx, data in enumerate(data_loader):
        tweet, user_h, ment_h, target = [item.to(device) for item in data]
        tweet, user_h, ment_h = map(embeddings, [tweet, user_h, ment_h])

        user_h = user_h.view(-1, 200, 33, 300)
        ment_h = ment_h.view(-1, 200, 33, 300)

        author_agent.reset_buffer()
        ment_agent.reset_buffer()

        output = classifier(tweet, user_h, ment_h, author_agent, ment_agent)
        output = output.squeeze(dim=-1)
        batch_acc = output.ge(0).float() == target
        acc_num.append(
            torch.sum(batch_acc.long()).item())
        total_num += len(target)

        py = output.data
        y_pred.append(py)
        y_test.append(target)

        print("Test: [{}/{}, {:.2f}%], acc: {:.4f}".format(
            batch_idx+1, len(data_loader),
            100 * (batch_idx+1) / len(data_loader), sum(acc_num)/total_num),
            end='\r')
    print()
    mrr = mrr_score(y_test, y_pred)
    bp = bpref(y_test, y_pred)
    print("MRR: {}, Bpref: {}".format(mrr, bp))

    precision = precision_score(y_test, y_pred, k=1)
    recall = recall_score(y_test, y_pred, k=1)
    hscore = hits_score(y_test, y_pred, k=1)
    hits3 = hits_score(y_test, y_pred, k=3)
    hits5 = hits_score(y_test, y_pred, k=5)
    F1 = 2 * (precision * recall) / (precision + recall)
    print(result_log.format(1, precision, recall, F1, hscore, hits3, hits5))

    return F1
