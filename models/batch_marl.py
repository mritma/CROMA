import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import torch.optim as optim
import numpy as np
import logging
import json

from models.classifier import TransformerClassifier, SingleTextClassifier
from models.classifier import TextEncoder, NoIterClassifier
from models.embedNet import EmbedNet

# from classifier import DotAttention, SingleClassifier
# from classifier import AddAttention
# from embedNet import EmbedNet


# torch.manual_seed(1)
# torch.cuda.manual_seed(1)

# TODO 载入模型，训练代码
# baseline: NB, PMPR, CAR


class Buffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def append(
        self, user_s, ment_s, user_a, ment_a, r, user_ab, ment_ab,
        user_next_s, ment_next_s, opposed_user_a, opposed_ment_a, label
    ):
        user_s, ment_s, user_a, ment_a, r, user_ab, ment_ab, \
            user_next_s, ment_next_s, \
            opposed_user_a, opposed_ment_a, label = map(
                lambda x: torch.split(x, 1, dim=0),
                [user_s, ment_s, user_a, ment_a, r, user_ab, ment_ab,
                 user_next_s, ment_next_s,
                 opposed_user_a, opposed_ment_a, label]
            )

        for i in range(len(r)):
            transition = (
                user_s[i], ment_s[i], user_a[i], ment_a[i], r[i],
                user_ab[i], ment_ab[i], user_next_s[i], ment_next_s[i],
                opposed_user_a[i], opposed_ment_a[i], label[i]
            )
            self.buffer.append(transition)

    def sample(self, batch_size, device):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        batch_user_s = [t[0] for t in batch]
        batch_ment_s = [t[1] for t in batch]
        batch_user_a = [t[2] for t in batch]
        batch_ment_a = [t[3] for t in batch]
        batch_r = [t[4] for t in batch]
        batch_user_ab = [t[5] for t in batch]
        batch_ment_ab = [t[6] for t in batch]
        batch_user_next_s = [t[7] for t in batch]
        batch_ment_next_s = [t[8] for t in batch]
        batch_opposed_user_a = [t[9] for t in batch]
        batch_opposed_ment_a = [t[10] for t in batch]
        batch_label = [t[11] for t in batch]

        batch_user_s = torch.cat(batch_user_s, dim=0)
        batch_ment_s = torch.cat(batch_ment_s, dim=0)
        batch_user_a = torch.cat(batch_user_a, dim=0)
        batch_ment_a = torch.cat(batch_ment_a, dim=0)
        batch_r = torch.cat(batch_r, dim=0)
        batch_r = batch_r.unsqueeze(dim=1)
        batch_user_ab = torch.cat(batch_user_ab, dim=0).squeeze(1)
        batch_ment_ab = torch.cat(batch_ment_ab, dim=0).squeeze(1)
        batch_user_next_s = torch.cat(batch_user_next_s, dim=0)
        batch_ment_next_s = torch.cat(batch_ment_next_s, dim=0)
        batch_opposed_user_a = torch.cat(batch_opposed_user_a, dim=0)
        batch_opposed_ment_a = torch.cat(batch_opposed_ment_a, dim=0)
        batch_label = torch.cat(batch_label, dim=0)

        return batch_user_s, batch_ment_s, batch_user_a, batch_ment_a, \
            batch_r, batch_user_ab, batch_ment_ab,\
            batch_user_next_s, batch_ment_next_s, \
            batch_opposed_user_a, batch_opposed_ment_a, batch_label

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


class BaseActor(nn.Module):
    def __init__(self, batch_size, history_num, device):
        super(BaseActor, self).__init__()
        self.selected_data = torch.zeros(batch_size, history_num).to(device)
        # (b, 50)
        self.unselected_data = \
            torch.zeros(batch_size, history_num).to(device)

    def forward(self, data, need_backward, is_target):
        raise NotImplementedError

    def choose_action(self, s, device, is_training):
        probs = self.map(s)
        # (b, 2)
        if is_training:
            # if random.uniform(0, 1) < 0.5:
            #     probs = probs * 0 + 0.5
            sampler = Categorical(probs)
            action = sampler.sample()
            # (b, )
        else:
            action = torch.max(probs, 1, keepdim=False)[1]
            # (b, )

        actions = action.unsqueeze(1)
        # (b, 1)
        batch_action = torch.zeros(s.size(0), 2).to(device)
        batch_action = batch_action.scatter_(1, actions, 1)
        return batch_action, actions
        # (b, 2)

    def get_log_probs(self, s, action):
        probs = self.map(s)
        # (b, 2)
        sampler = Categorical(probs)
        log_probs = sampler.log_prob(action)
        # TODO 此处重大修改
        # action = action.byte()
        # probs = torch.masked_select(probs, action)
        # log_probs = torch.log(probs+1e-25)
        return log_probs


class MLPActor(BaseActor):
    def __init__(self, batch_size, history_num, d_embed, device):
        super(MLPActor, self).__init__(batch_size, history_num, device)
        self.map = nn.Sequential(
            # nn.Tanh(),
            nn.Linear(d_embed*7, d_embed*2),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(d_embed*2, d_embed),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(d_embed, 2),
            nn.Softmax(dim=-1))

    def forward(self, g_state, history, step, device, is_training, is_target):
        with torch.no_grad():
            state = torch.cat(
                [g_state, history[:, step]], dim=-1
            )
            # (b, d_embed*7)
            action, b_action = self.choose_action(state, device, is_training)
            # (b, 2), (b, 1)
            # target actor cannot update the data buffer
            if not is_target:
                b_action = b_action.squeeze(1)
                self.selected_data[:, step] = b_action
                self.unselected_data[:, step] = 1 - b_action
                b_action = b_action.unsqueeze(1)

            return state, action, b_action

    def select_history(self, history, mask):
        selected_h = history * mask
        sum_h = selected_h.sum(dim=1)
        len_h = mask.sum(dim=1) + 1e-25
        state = sum_h / len_h

        return state

    def get_sub_state(self, history):
        state_a = self.select_history(
            history, self.selected_data.unsqueeze(-1)
        )
        # (b, d_embed)
        state_b = self.select_history(
            history, self.unselected_data.unsqueeze(-1)
        )
        # (b, d_embed)
        state = torch.cat(
            [state_a, state_b], dim=-1
        )

        return state


class Critic(nn.Module):
    # TODO 改为同时输出所有动作的Q
    def __init__(self, text_state_dim, encoding_dim):
        super(Critic, self).__init__()
        self.QUser = nn.Sequential(
            nn.Linear(text_state_dim + 2, text_state_dim),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(text_state_dim, encoding_dim),
            nn.Tanh()
        )
        self.QMent = nn.Sequential(
            nn.Linear(text_state_dim + 2, text_state_dim),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(text_state_dim, encoding_dim),
            nn.Tanh()
        )
        self.QScore = nn.Sequential(
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(encoding_dim, 1)
        )

    def forward(self, user_s, user_a, ment_s, ment_a):
        user_rep = self.QUser(torch.cat([user_s, user_a], dim=-1))
        ment_rep = self.QMent(torch.cat([ment_s, ment_a], dim=-1))
        score = self.QScore(torch.cat([user_rep, ment_rep], dim=-1))
        return score


def reward_shaping(batch_r):
    # batch_mean = torch.mean(batch_r)
    # batch_std = torch.std(batch_r)
    # new_r = (batch_r - batch_mean) / batch_std
    new_r = batch_r * 10

    return new_r


class BatchMultiAgentClassifier():
    def __init__(self, n_vocab, d_embed, batch_size, history_num, device):
        self.history_num = history_num

        self.embedding = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)

        self.textEncoder = TextEncoder()
        self.user_actor = MLPActor(batch_size, history_num, d_embed, device)
        self.ment_actor = MLPActor(batch_size, history_num, d_embed, device)
        self.target_user_actor = MLPActor(
            batch_size, history_num, d_embed, device
        )
        self.target_ment_actor = MLPActor(
            batch_size, history_num, d_embed, device
        )
        self.critic = Critic(text_state_dim=d_embed*7,
                             encoding_dim=d_embed)
        self.target_critic = Critic(text_state_dim=d_embed*7,
                                    encoding_dim=d_embed)
        self.classifier = SingleTextClassifier(d_embed)
        # self.classifier = NoIterClassifier(d_embed)

        # self.load_pre_trained()
        self.embedding.to(device)
        self.user_embedding_a.to(device)
        self.user_embedding_b.to(device)
        self.ment_embedding_a.to(device)
        self.ment_embedding_b.to(device)
        self.textEncoder.to(device)
        self.user_actor.to(device)
        self.ment_actor.to(device)
        self.target_user_actor.to(device)
        self.target_ment_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.classifier.to(device)

        self.user_actor_optimizer = optim.Adam(self.user_actor.parameters(),
                                               lr=0.001, weight_decay=1e-6)
        self.ment_actor_optimizer = optim.Adam(self.ment_actor.parameters(),
                                               lr=0.001, weight_decay=1e-6)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=0.001, weight_decay=1e-6)
        self.classifier_optimizer = optim.Adam(
            [{"params": self.classifier.parameters()},
             {"params": self.embedding.parameters()},
             {"params": self.user_embedding_a.parameters()},
             {"params": self.user_embedding_b.parameters()},
             {"params": self.ment_embedding_a.parameters()},
             {"params": self.ment_embedding_b.parameters()}])

        self.buffer = Buffer(max_len=100000)
        # bind target buffer to source buffer, for list is a reference type
        self.target_user_actor.selected_data = self.user_actor.selected_data
        self.target_user_actor.unselected_data = \
            self.user_actor.unselected_data
        self.target_ment_actor.selected_data = self.ment_actor.selected_data
        self.target_ment_actor.unselected_data = \
            self.ment_actor.unselected_data
        self.hard_update()

    def update_buffer(self, data, device,
                      need_backward=True, train_classifier=True,
                      update_buffer=True):
        tweet, user_hs, ment_hs, ground_truth = \
            [item.to(device) for item in data]
        # user_hs: (batch, 33*50), batch size fixed to 1

        if train_classifier:
            self.embedding.train()
            self.user_embedding_a.train()
            self.user_embedding_b.train()
            self.ment_embedding_a.train()
            self.ment_embedding_b.train()
            self.classifier.train()
        else:
            self.embedding.eval()
            self.user_embedding_a.eval()
            self.user_embedding_b.eval()
            self.ment_embedding_a.eval()
            self.ment_embedding_b.eval()

        tweet = self.embedding(tweet)
        user_hs_a = self.user_embedding_a(user_hs)
        user_hs_b = self.user_embedding_b(user_hs)
        ment_hs_a = self.ment_embedding_a(ment_hs)
        ment_hs_b = self.ment_embedding_b(ment_hs)
        # user_hs: (batch, 33*50, 200)

        user_hs_a, user_hs_b, ment_hs_a, ment_hs_b = map(
            lambda x: x.view(-1, self.history_num, 33, 200),
            [user_hs_a, user_hs_b, ment_hs_a, ment_hs_b])
        tweet = torch.sum(tweet, dim=-2)
        user_hs = self.textEncoder(tweet, user_hs_a, user_hs_b)
        ment_hs = self.textEncoder(tweet, ment_hs_a, ment_hs_b)
        # user_hs: (batch, 50, 200)

        tweet_num = user_hs.size(1)
        label = ground_truth
        output = None

        batch_size = user_hs.size(0)
        history_num = user_hs.size(1)

        assert history_num == self.history_num
        self.reset_actor_mask(batch_size, history_num, device)

        _, global_hidden, _ = self.classifier(
            tweet, user_hs, ment_hs
        )

        origin_tweet = tweet

        with torch.no_grad():

            user_mask = self.user_actor.selected_data
            ment_mask = self.ment_actor.selected_data
            output, local_hidden, _ = self.classifier(
                origin_tweet, user_hs, ment_hs, user_mask, ment_mask)
            prob = torch.sigmoid(output)
            prob = prob.squeeze(dim=-1)
            last_score = label * prob + (1-label) * (1 - prob)
            # output = output.squeeze(1)
            # last_score = F.binary_cross_entropy_with_logits(
            #     output, ground_truth,
            #     reduction='none'
            # )

        # last_score = 0.5

        select_index = list(range(tweet_num - 1))
        for i in range(tweet_num-1):
            with torch.no_grad():
                user_sub_state = self.user_actor.get_sub_state(user_hs)
                # (b, embed*2)
                ment_sub_state = self.ment_actor.get_sub_state(ment_hs)
                g_state = torch.cat(
                    [user_sub_state, ment_sub_state,
                     origin_tweet, global_hidden],
                    dim=-1
                )
                # (b, embed*5)
                user_s1, user_a1, user_ab = self.user_actor(
                    g_state, user_hs, i, device,
                    is_training=need_backward, is_target=False
                )
                ment_s1, ment_a1, ment_ab = self.ment_actor(
                    g_state, ment_hs, i, device,
                    is_training=need_backward, is_target=False
                )

                # user_mask = self.user_actor.selected_data
                # ment_mask = self.ment_actor.selected_data
                # output, local_hidden = self.classifier(
                #     origin_tweet, user_hs, ment_hs, user_mask, ment_mask)

                if i in select_index and update_buffer is True:
                    user_mask = self.user_actor.selected_data
                    ment_mask = self.ment_actor.selected_data
                    output, _, _ = self.classifier(
                        origin_tweet, user_hs, ment_hs, user_mask, ment_mask)
                    prob = torch.sigmoid(output)
                    prob = prob.squeeze(dim=-1)
                    score = label * prob + (1-label) * (1 - prob)
                    r_1 = score - last_score
                    # output = output.squeeze(1)
                    # score = F.binary_cross_entropy_with_logits(
                    #     output, ground_truth,
                    #     reduction='none'
                    # )
                    # r_1 = last_score - score
                    # r = (29 * label + 1) * r_1
                    r = r_1

                    last_score = score
                    user_sub_state = self.user_actor.get_sub_state(user_hs)
                    # (b, embed*2)
                    ment_sub_state = self.ment_actor.get_sub_state(ment_hs)
                    g_state = torch.cat(
                        [user_sub_state, ment_sub_state,
                         origin_tweet, global_hidden],
                        dim=-1
                    )
                    # (b, embed*5)
                    user_s2, _, _ = self.target_user_actor(
                        g_state, user_hs, i+1, device,
                        is_training=False, is_target=True
                    )
                    ment_s2, _, _ = self.target_ment_actor(
                        g_state, ment_hs, i+1, device,
                        is_training=False, is_target=True
                    )

                    opposed_user_a = 1 - user_a1
                    opposed_ment_a = 1 - ment_a1
                    # state: (b, d*6), action: (b, 2), r: (b, 1)

                    if update_buffer:
                        self.buffer.append(
                            user_s1, ment_s1, user_a1, ment_a1,
                            r, user_ab, ment_ab,
                            user_s2, ment_s2, opposed_user_a, opposed_ment_a,
                            label
                        )

        user_mask = self.user_actor.selected_data
        ment_mask = self.ment_actor.selected_data

        selected_num = (
            user_mask.sum(dim=-1).mean(),
            ment_mask.sum(dim=-1).mean()
        )

        output, _, _ = self.classifier(
            origin_tweet, user_hs, ment_hs, user_mask, ment_mask
        )
        output = output.squeeze(dim=-1)
        c_loss = F.binary_cross_entropy_with_logits(output, ground_truth)

        with torch.no_grad():
            batch_acc = output.ge(0).float() == ground_truth
            predict = output.ge(0).float()

        if train_classifier:
            self.classifier_optimizer.zero_grad()
            c_loss.backward()
            self.classifier_optimizer.step()

        return c_loss.item(), output.detach(), batch_acc, \
            ground_truth, selected_num, predict

    def optimize_AC(self, device, weight=0):
        # update critic
        batch_user_s, batch_ment_s, batch_user_a, batch_ment_a, batch_r, \
            batch_user_ab, batch_ment_ab, \
            batch_user_next_s, batch_ment_next_s, batch_opposed_user_a, \
            batch_opposed_ment_a, batch_label = self.buffer.sample(
                1024, device
            )

        batch_r = reward_shaping(batch_r)

        Q_predicted = self.critic(
            batch_user_s, batch_user_a, batch_ment_s, batch_ment_a)

        with torch.no_grad():
            batch_user_next_a, _ = self.target_user_actor.choose_action(
                batch_user_next_s, device, is_training=False)
            batch_ment_next_a, _ = self.target_ment_actor.choose_action(
                batch_ment_next_s, device, is_training=False)

            Q2 = self.target_critic(
                batch_user_next_s, batch_user_next_a,
                batch_ment_next_s, batch_ment_next_a)
            Q_expected = batch_r + 0.99 * Q2

        loss_critic = F.smooth_l1_loss(Q_predicted, Q_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        self.soft_update(0.0001, u_type='critic')
        # update actor
        with torch.no_grad():
            user_A = self.target_critic(
                batch_user_s, batch_user_a, batch_ment_s, batch_ment_a) \
                - self.target_critic(
                    batch_user_s, batch_opposed_user_a,
                    batch_ment_s, batch_ment_a)
            ment_A = self.target_critic(
                batch_user_s, batch_user_a, batch_ment_s, batch_ment_a) \
                - self.target_critic(
                    batch_user_s, batch_user_a,
                    batch_ment_s, batch_opposed_ment_a)
            # user_A = self.target_critic(
            #     batch_user_s, batch_user_a,
            #     batch_ment_s, batch_ment_a
            # )
            # ment_A = self.target_critic(
            #     batch_user_s, batch_user_a,
            #     batch_ment_s, batch_ment_a
            # )
            user_A = user_A.squeeze(1)
            ment_A = ment_A.squeeze(1)

        user_log_probs = self.user_actor.get_log_probs(
            batch_user_s, batch_user_ab)
        ment_log_probs = self.ment_actor.get_log_probs(
            batch_ment_s, batch_ment_ab)
        sample_weight = 1 + batch_label * 49
        loss_u_actor = -torch.mean(
            user_log_probs * (user_A - weight * user_log_probs)
        )
        loss_m_actor = -torch.mean(
            ment_log_probs * (ment_A - weight * ment_log_probs)
        )
        self.user_actor_optimizer.zero_grad()
        self.ment_actor_optimizer.zero_grad()
        loss_u_actor.backward()
        loss_m_actor.backward()
        self.user_actor_optimizer.step()
        self.ment_actor_optimizer.step()
        self.soft_update(0.1, u_type='actor')
        return loss_critic.item(), loss_u_actor.item(), loss_m_actor.item(), user_A.mean().item(), ment_A.mean().item()

    def reset_actor_mask(self, batch_size, history_num, device):
        self.user_actor.selected_data = torch.zeros(
            batch_size, history_num).to(device)
        self.user_actor.unselected_data = torch.zeros(
            batch_size, history_num).to(device)
        self.ment_actor.selected_data = torch.zeros(
            batch_size, history_num).to(device)
        self.ment_actor.unselected_data = torch.zeros(
            batch_size, history_num).to(device)

        self.target_user_actor.selected_data = self.user_actor.selected_data
        self.target_user_actor.unselected_data = \
            self.user_actor.unselected_data
        self.target_ment_actor.selected_data = self.ment_actor.selected_data
        self.target_ment_actor.unselected_data = \
            self.ment_actor.unselected_data

    def hard_update(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_user_actor.load_state_dict(self.user_actor.state_dict())
        self.target_ment_actor.load_state_dict(self.ment_actor.state_dict())

    def soft_update(self, scale, u_type):
        if u_type == 'critic':
            for target_param, param in zip(self.target_critic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(
                    param.data * scale + target_param.data * (1 - scale))
        if u_type == 'actor':
            for target_param, param in zip(self.target_user_actor.parameters(),
                                           self.user_actor.parameters()):
                # target_param.data.copy_(target_param.data)
                target_param.data.copy_(
                    param.data * scale + target_param.data * (1 - scale))
            for target_param, param in zip(self.target_ment_actor.parameters(),
                                           self.ment_actor.parameters()):
                # target_param.data.copy_(target_param.data)
                target_param.data.copy_(
                    param.data * scale + target_param.data * (1 - scale))
        if u_type == 'all':
            for target_param, param in zip(self.target_critic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(
                    param.data * scale + target_param.data * (1 - scale))
            for target_param, param in zip(self.target_user_actor.parameters(),
                                           self.user_actor.parameters()):
                target_param.data.copy_(target_param.data)
            for target_param, param in zip(self.target_ment_actor.parameters(),
                                           self.ment_actor.parameters()):
                target_param.data.copy_(target_param.data)

    def pre_train_classifier(self, data, device):
        # train one batch
        self.embedding.train()
        self.user_embedding_a.train()
        self.user_embedding_b.train()
        self.ment_embedding_a.train()
        self.ment_embedding_b.train()
        self.classifier.train()

        tweet, user_hs, ment_hs, ground_truth = \
            [item.to(device) for item in data]
        # user_hs: (batch, 33*50), batch size fixed to 1

        tweet = self.embedding(tweet)
        user_hs_a = self.user_embedding_a(user_hs)
        user_hs_b = self.user_embedding_b(user_hs)
        ment_hs_a = self.ment_embedding_a(ment_hs)
        ment_hs_b = self.ment_embedding_b(ment_hs)
        # user_hs: (batch, 33*50, 200)

        user_hs_a, user_hs_b, ment_hs_a, ment_hs_b = map(
            lambda x: x.view(-1, self.history_num, 33, 200),
            [user_hs_a, user_hs_b, ment_hs_a, ment_hs_b])
        tweet = torch.sum(tweet, dim=-2)
        user_hs = self.textEncoder(tweet, user_hs_a, user_hs_b)
        ment_hs = self.textEncoder(tweet, ment_hs_a, ment_hs_b)
        # user_hs: (batch, 50, 200)

        output, _, _ = self.classifier(tweet, user_hs, ment_hs)
        output = output.squeeze(dim=-1)

        c_loss = F.binary_cross_entropy_with_logits(
            output, ground_truth
        )

        batch_acc = output.ge(0).float() == ground_truth
        predict = output.ge(0).float()

        self.classifier_optimizer.zero_grad()
        c_loss.backward()
        self.classifier_optimizer.step()

        return c_loss.item(), output, batch_acc, ground_truth, predict

    def joint_train(self, data, device):
        cf_loss, _, batch_acc, _ = self.update_buffer(
            data, device, need_backward=True,
            train_classifier=True, update_buffer=True)
        self.optimize_AC(device)
        return cf_loss, batch_acc

    def to(self, device):
        self.embedding.to(device)
        self.user_embedding_a.to(device)
        self.user_embedding_b.to(device)
        self.ment_embedding_a.to(device)
        self.ment_embedding_b.to(device)
        self.textEncoder.to(device)
        self.user_actor.to(device)
        self.ment_actor.to(device)
        self.target_user_actor.to(device)
        self.target_ment_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.classifier.to(device)

    def load_pre_trained(self):

        self.embedding.load_state_dict(
            torch.load("parameter/1w3s_pos10/embedding"))
        self.user_embedding_a.load_state_dict(
            torch.load("parameter/1w3s_pos10/user_embedding_a"))
        self.user_embedding_b.load_state_dict(
            torch.load("parameter/1w3s_pos10/user_embedding_b"))
        self.ment_embedding_a.load_state_dict(
            torch.load("parameter/1w3s_pos10/ment_embedding_a"))
        self.ment_embedding_b.load_state_dict(
            torch.load("parameter/1w3s_pos10/ment_embedding_b"))

        self.textEncoder.load_state_dict(
            torch.load("parameter/1w3s_pos10/text_encoder")
        )
        self.classifier.load_state_dict(
            torch.load("parameter/1w3s_pos10/classifier")
        )

    def classifier_benchmark(self, data, device):
        tweet, user_hs, ment_hs, ground_truth = \
            [item.to(device) for item in data]
        # user_hs: (batch, 33*50), batch size fixed to 1

        self.embedding = self.embedding.eval()
        self.user_embedding_a = self.user_embedding_a.eval()
        self.user_embedding_b = self.user_embedding_b.eval()
        self.ment_embedding_a = self.ment_embedding_a.eval()
        self.ment_embedding_b = self.ment_embedding_b.eval()

        with torch.no_grad():
            tweet = self.embedding(tweet)
            user_hs_a = self.user_embedding_a(user_hs)
            user_hs_b = self.user_embedding_b(user_hs)
            ment_hs_a = self.ment_embedding_a(ment_hs)
            ment_hs_b = self.ment_embedding_b(ment_hs)
            # user_hs: (batch, 33*50, 200)

            user_hs_a, user_hs_b, ment_hs_a, ment_hs_b = map(
                lambda x: x.view(-1, self.history_num, 33, 200),
                [user_hs_a, user_hs_b, ment_hs_a, ment_hs_b])
            tweet = torch.sum(tweet, dim=-2)
            user_hs = self.textEncoder(tweet, user_hs_a, user_hs_b)
            ment_hs = self.textEncoder(tweet, ment_hs_a, ment_hs_b)
            # user_hs: (batch, 50, 200)

            output, _, _ = self.classifier(tweet, user_hs, ment_hs)
            output = output.squeeze(dim=-1)

            c_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
            batch_acc = output.ge(0).float() == ground_truth
            predict = output.ge(0).float()

        return c_loss.item(), output, batch_acc, ground_truth, predict

    def load_state_dict(self, state_dict):
        self.embedding.load_state_dict(state_dict["embedding"])
        self.user_embedding_a.load_state_dict(state_dict["user_embedding_a"])
        self.user_embedding_b.load_state_dict(state_dict["user_embedding_b"])
        self.ment_embedding_a.load_state_dict(state_dict["ment_embedding_a"])
        self.ment_embedding_b.load_state_dict(state_dict["ment_embedding_b"])
        self.textEncoder.load_state_dict(state_dict["text_encoder"])
        self.user_actor.load_state_dict(state_dict["user_actor"])
        self.ment_actor.load_state_dict(state_dict["ment_actor"])
        self.target_user_actor.load_state_dict(state_dict["target_user_actor"])
        self.target_ment_actor.load_state_dict(state_dict["target_ment_actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.classifier.load_state_dict(state_dict["classifier"])

    def load_pre_trained_rl(self, state_dict):
        self.user_actor.load_state_dict(state_dict["user_actor"])
        self.ment_actor.load_state_dict(state_dict["ment_actor"])
        self.target_user_actor.load_state_dict(state_dict["target_user_actor"])
        self.target_ment_actor.load_state_dict(state_dict["target_ment_actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])

    def modify_classifier_lr(self, lr):
        for param_group in self.classifier_optimizer.param_groups:
            param_group['lr'] = lr
        logging.info(f"Set classifier lr to {lr}")

    def modify_rl_lr(self, lr):
        for param_group in self.user_actor_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.ment_actor_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr
        logging.info(f"Set actor and critic lr to {lr}")

    def select_sentence(
        self, data, device
    ):
        tweet, user_hs, ment_hs, ground_truth = \
            [item.to(device) for item in data]
        # user_hs: (batch, 33*50), batch size fixed to 1

        self.embedding.eval()
        self.user_embedding_a.eval()
        self.user_embedding_b.eval()
        self.ment_embedding_a.eval()
        self.ment_embedding_b.eval()
        self.classifier.eval()

        with torch.no_grad():
            tweet = self.embedding(tweet)
            user_hs_a = self.user_embedding_a(user_hs)
            user_hs_b = self.user_embedding_b(user_hs)
            ment_hs_a = self.ment_embedding_a(ment_hs)
            ment_hs_b = self.ment_embedding_b(ment_hs)
            # user_hs: (batch, 33*50, 200)

            user_hs_a, user_hs_b, ment_hs_a, ment_hs_b = map(
                lambda x: x.view(-1, self.history_num, 33, 200),
                [user_hs_a, user_hs_b, ment_hs_a, ment_hs_b])
            tweet = torch.sum(tweet, dim=-2)
            user_hs = self.textEncoder(tweet, user_hs_a, user_hs_b)
            ment_hs = self.textEncoder(tweet, ment_hs_a, ment_hs_b)
            # user_hs: (batch, 50, 200)

            tweet_num = user_hs.size(1)

            batch_size = user_hs.size(0)
            history_num = user_hs.size(1)

            assert history_num == self.history_num
            self.reset_actor_mask(batch_size, history_num, device)

            _, global_hidden, _ = self.classifier(
                tweet, user_hs, ment_hs
            )

            origin_tweet = tweet

            for i in range(tweet_num-1):
                user_sub_state = self.user_actor.get_sub_state(user_hs)
                # (b, embed*2)
                ment_sub_state = self.ment_actor.get_sub_state(ment_hs)
                g_state = torch.cat(
                    [user_sub_state, ment_sub_state,
                     tweet, global_hidden],
                    dim=-1
                )
                # (b, embed*5)
                self.user_actor(
                    g_state, user_hs, i, device,
                    is_training=False, is_target=False
                )
                self.ment_actor(
                    g_state, ment_hs, i, device,
                    is_training=False, is_target=False
                )

        user_mask = self.user_actor.selected_data
        ment_mask = self.ment_actor.selected_data

        return user_mask, ment_mask

    def find_indicator(
        self, data, device
    ):
        flag = False
        indicator = dict()
        tweet, user_hs, ment_hs, ground_truth = \
            [item.to(device) for item in data]
        # user_hs: (batch, 33*50), batch size fixed to 1

        self.embedding.eval()
        self.user_embedding_a.eval()
        self.user_embedding_b.eval()
        self.ment_embedding_a.eval()
        self.ment_embedding_b.eval()
        self.classifier.eval()

        with torch.no_grad():
            tweet = self.embedding(tweet)
            user_hs_a = self.user_embedding_a(user_hs)
            user_hs_b = self.user_embedding_b(user_hs)
            ment_hs_a = self.ment_embedding_a(ment_hs)
            ment_hs_b = self.ment_embedding_b(ment_hs)
            # user_hs: (batch, 33*50, 200)

            user_hs_a, user_hs_b, ment_hs_a, ment_hs_b = map(
                lambda x: x.view(-1, self.history_num, 33, 200),
                [user_hs_a, user_hs_b, ment_hs_a, ment_hs_b])
            tweet = torch.sum(tweet, dim=-2)
            user_hs = self.textEncoder(tweet, user_hs_a, user_hs_b)
            ment_hs = self.textEncoder(tweet, ment_hs_a, ment_hs_b)
            # user_hs: (batch, 50, 200)

            tweet_num = user_hs.size(1)

            batch_size = user_hs.size(0)
            history_num = user_hs.size(1)

            assert history_num == self.history_num
            self.reset_actor_mask(batch_size, history_num, device)

            old_logits, global_hidden, attn_weight = self.classifier(
                tweet, user_hs, ment_hs
            )

            origin_tweet = tweet

            for i in range(tweet_num-1):
                user_sub_state = self.user_actor.get_sub_state(user_hs)
                # (b, embed*2)
                ment_sub_state = self.ment_actor.get_sub_state(ment_hs)
                g_state = torch.cat(
                    [user_sub_state, ment_sub_state,
                     tweet, global_hidden],
                    dim=-1
                )
                # (b, embed*5)
                self.user_actor(
                    g_state, user_hs, i, device,
                    is_training=False, is_target=False
                )
                self.ment_actor(
                    g_state, ment_hs, i, device,
                    is_training=False, is_target=False
                )

        user_mask = self.user_actor.selected_data
        ment_mask = self.ment_actor.selected_data

        new_logits, _, _ = self.classifier(
            origin_tweet, user_hs, ment_hs, user_mask, ment_mask
        )
        old_logits = old_logits.squeeze(1).item()
        new_logits = new_logits.squeeze(1).item()

        if new_logits > 0 and new_logits - old_logits > 3:
            indicator["new_logits"] = new_logits
            indicator["old_logits"] = old_logits
            indicator["user_mask"] = user_mask.unsqueeze(0).cpu().tolist()
            indicator["ment_mask"] = ment_mask.unsqueeze(0).cpu().tolist()
            u_attn, m_attn = attn_weight
            # attn_weight: (b, 50, 1)
            u_attn = u_attn.unsqueeze(-1)
            u_attn = u_attn.unsqueeze(0)
            m_attn = m_attn.unsqueeze(-1)
            m_attn = m_attn.unsqueeze(0)
            indicator["user_attn"] = u_attn.cpu().tolist()
            indicator["ment_attn"] = m_attn.cpu().tolist()
            flag = True

        return indicator, flag
