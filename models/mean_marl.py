import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import torch.optim as optim
import numpy as np


from models.classifier import *
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

    def append(self, user_s, ment_s, user_a, ment_a, r,
               user_next_s, ment_next_s, opposed_user_a, opposed_ment_a):
        user_s, ment_s, user_a, ment_a, r, \
            user_next_s, ment_next_s, \
            opposed_user_a, opposed_ment_a = map(
                lambda x: torch.split(x, 1, dim=0),
                [user_s, ment_s, user_a, ment_a, r, user_next_s,
                 ment_next_s, opposed_user_a, opposed_ment_a]
            )

        for i in range(len(r)):
            transition = (user_s[i], ment_s[i], user_a[i], ment_a[i], r[i],
                          user_next_s[i], ment_next_s[i],
                          opposed_user_a[i], opposed_ment_a[i])
            self.buffer.append(transition)

    def sample(self, batch_size, device):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        batch_user_s = [t[0] for t in batch]
        batch_ment_s = [t[1] for t in batch]
        batch_user_a = [t[2] for t in batch]
        batch_ment_a = [t[3] for t in batch]
        batch_r = [t[4] for t in batch]
        batch_user_next_s = [t[5] for t in batch]
        batch_ment_next_s = [t[6] for t in batch]
        batch_opposed_user_a = [t[7] for t in batch]
        batch_opposed_ment_a = [t[8] for t in batch]

        batch_user_s = torch.cat(batch_user_s, dim=0)
        batch_ment_s = torch.cat(batch_ment_s, dim=0)
        batch_user_a = torch.cat(batch_user_a, dim=0)
        batch_ment_a = torch.cat(batch_ment_a, dim=0)
        batch_r = torch.cat(batch_r, dim=0)
        batch_r = batch_r.unsqueeze(dim=1)
        batch_user_next_s = torch.cat(batch_user_next_s, dim=0)
        batch_ment_next_s = torch.cat(batch_ment_next_s, dim=0)
        batch_opposed_user_a = torch.cat(batch_opposed_user_a, dim=0)
        batch_opposed_ment_a = torch.cat(batch_opposed_ment_a, dim=0)

        return batch_user_s, batch_ment_s, batch_user_a, batch_ment_a, \
            batch_r, batch_user_next_s, batch_ment_next_s, \
            batch_opposed_user_a, batch_opposed_ment_a

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
            sampler = Categorical(probs)
            action = sampler.sample()
            # (b, )
        else:
            action = torch.max(probs, 1, keepdim=False)[1]
            # (b, )

        actions = action.unsqueeze(1)
        # (b, 1)
        return torch.zeros(s.size(0), 2).to(device).scatter_(1, actions, 1), actions
        # (b, 2)

    def get_log_probs(self, s, action):
        probs = self.map(s)
        # (b, 2)
        sampler = Categorical(probs)
        log_probs = sampler.log_prob(action)
        return log_probs


class MLPActor(BaseActor):
    def __init__(self, batch_size, history_num, d_embed, device):
        super(MLPActor, self).__init__(batch_size, history_num, device)
        self.map = nn.Sequential(
            nn.Linear(d_embed*10, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 2),
            nn.Softmax(dim=-1))

    def forward(self, g_state, history, step, device, is_training, is_target):
        with torch.no_grad():
            # state_a = self.get_state(
            #     history, self.selected_data.unsqueeze(-1)
            # )
            # # (b, d_embed)
            # state_b = self.get_state(
            #     history, self.unselected_data.unsqueeze(-1)
            # )
            # # (b, d_embed)
            # state = torch.cat(
            #     [state_a, state_b, tweet, history[:, step]], dim=-1
            # )
            # (b, d_embed*4)
            state = torch.cat(
                [g_state, history[:, step]], dim=-1
            )
            # (b, d_embed*10)
            action, b_action = self.choose_action(state, device, is_training)
            # (b, 2), (b, 1)
            # target actor cannot update the data buffer
            if not is_target:
                b_action = b_action.squeeze(1)
                self.selected_data[:, step] = b_action
                self.unselected_data[:, step] = 1 - b_action

            return state, action

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
        # (b, d_embed*2)
        state_b = self.select_history(
            history, self.unselected_data.unsqueeze(-1)
        )
        # (b, d_embed*2)
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
            nn.Linear(text_state_dim, encoding_dim),
            nn.Tanh()
        )
        self.QMent = nn.Sequential(
            nn.Linear(text_state_dim + 2, text_state_dim),
            nn.ReLU(),
            nn.Linear(text_state_dim, encoding_dim),
            nn.Tanh()
        )
        self.QScore = nn.Sequential(
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, 1)
        )

    def forward(self, user_s, user_a, ment_s, ment_a):
        user_rep = self.QUser(torch.cat([user_s, user_a], dim=-1))
        ment_rep = self.QMent(torch.cat([ment_s, ment_a], dim=-1))
        score = self.QScore(torch.cat([user_rep, ment_rep], dim=-1))
        return score


class MeanMultiAgentClassifier():
    def __init__(self, n_vocab, d_embed, batch_size, history_num, device, base):
        if base == "dan":
            self.text_encoder = MixDANEncoder(n_vocab, d_embed)
            self.classifier = MLPClassifier(d_embed*2, 1)
        elif base == "gru":
            self.text_encoder = GRUEncoder(n_vocab, d_embed)
            self.classifier = MLPClassifier(d_embed*3, 1)
        self.base = base

        self.user_actor = MLPActor(batch_size, history_num, d_embed, device)
        self.ment_actor = MLPActor(batch_size, history_num, d_embed, device)
        self.target_user_actor = MLPActor(
            batch_size, history_num, d_embed, device
        )
        self.target_ment_actor = MLPActor(
            batch_size, history_num, d_embed, device
        )
        self.critic = Critic(text_state_dim=d_embed*10,
                             encoding_dim=d_embed)
        self.target_critic = Critic(text_state_dim=d_embed*10,
                                    encoding_dim=d_embed)

        self.load_pre_trained(base)
        self.to(device)

        self.user_actor_optimizer = optim.Adam(self.user_actor.parameters(),
                                               lr=0.001, weight_decay=1e-6)
        self.ment_actor_optimizer = optim.Adam(self.ment_actor.parameters(),
                                               lr=0.001, weight_decay=1e-6)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=0.001, weight_decay=1e-6)
        self.classifier_optimizer = optim.Adam(
            [{"params": self.classifier.parameters()},
             {"params": self.text_encoder.parameters()}])

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
        last_score = 0.5
        tweet, user_hs, ment_hs, ground_truth = \
            [item.to(device) for item in data]
        # user_hs: (batch, 33*50)

        if train_classifier:
            self.text_encoder.train()
            self.classifier.train()
        else:
            self.text_encoder.eval()
            self.classifier.eval()

        user_hs, ment_hs = self.text_encoder(tweet, user_hs, ment_hs)
        # user_hs: (batch, 50, d_embed*2)
        tweet_num = user_hs.size(1)
        label = ground_truth
        # TODO is label useful?
        output = None

        batch_size = user_hs.size(0)
        history_num = user_hs.size(1)
        assert history_num == 50
        self.reset_actor_mask(batch_size, history_num, device)

        select_index = list(np.random.randint(1, tweet_num, 3))
        # select_index = list(range(tweet_num - 1))
        for i in range(tweet_num-1):
            with torch.no_grad():
                user_sub_state = self.user_actor.get_sub_state(user_hs)
                # (b, embed*4)
                ment_sub_state = self.ment_actor.get_sub_state(ment_hs)
                g_state = torch.cat(
                    [user_sub_state, ment_sub_state], dim=-1
                )
                # (b, embed*8)
                user_s1, user_a1 = self.user_actor(g_state, user_hs, i,
                                                   device,
                                                   is_training=need_backward,
                                                   is_target=False)
                ment_s1, ment_a1 = self.ment_actor(g_state, ment_hs, i,
                                                   device,
                                                   is_training=need_backward,
                                                   is_target=False)

                if i in select_index and update_buffer is True:
                    user_mask = self.user_actor.selected_data
                    ment_mask = self.ment_actor.selected_data
                    mask = torch.cat([user_mask, ment_mask], dim=-1)
                    # mask (b, 100)
                    history = torch.cat([user_hs, ment_hs], dim=1)
                    # history (b, 100, d_embed*2)
                    mask = mask.unsqueeze(dim=-1)
                    history = history * mask
                    history = history.sum(dim=1)
                    # (b, d_embed*2)
                    len_h = mask.sum(dim=1) + 1e-25
                    # len_h (b, 1)
                    history = history / len_h
                    # (b, d_embed*2)

                    output = self.classifier(history)
                    prob = torch.sigmoid(output)
                    prob = prob.squeeze(dim=-1)
                    score = label * prob + (1-label) * (1 - prob)
                    r = score - last_score

                    last_score = score
                    user_sub_state = self.user_actor.get_sub_state(user_hs)
                    # (b, embed*4)
                    ment_sub_state = self.ment_actor.get_sub_state(ment_hs)
                    g_state = torch.cat(
                        [user_sub_state, ment_sub_state], dim=-1
                    )
                    # (b, embed*8)
                    user_s2, _ = self.target_user_actor(g_state, user_hs, i+1,
                                                        device,
                                                        is_training=False,
                                                        is_target=True)
                    ment_s2, _ = self.target_ment_actor(g_state, ment_hs, i+1,
                                                        device,
                                                        is_training=False,
                                                        is_target=True)

                    opposed_user_a = 1 - user_a1
                    opposed_ment_a = 1 - ment_a1
                    # state: (b, d*10), action: (b, 2), r: (b, 1)
                    if update_buffer:
                        self.buffer.append(
                            user_s1, ment_s1, user_a1, ment_a1, r,
                            user_s2, ment_s2, opposed_user_a, opposed_ment_a)

        user_mask = self.user_actor.selected_data
        ment_mask = self.ment_actor.selected_data

        selected_num = (
            user_mask.sum(dim=-1).mean(),
            ment_mask.sum(dim=-1).mean()
        )

        mask = torch.cat([user_mask, ment_mask], dim=-1)
        # mask (b, 100)
        history = torch.cat([user_hs, ment_hs], dim=1)
        # history (b, 100, d_embed*2)
        mask = mask.unsqueeze(dim=-1)
        history = history * mask
        history = history.sum(dim=1)
        # (b, d_embed*2)
        len_h = mask.sum(dim=1) + 1e-25
        # len_h (b, 1)
        history = history / len_h
        # (b, d_embed*2)

        output = self.classifier(history)
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

    def optimize_AC(self, device):
        # update critic
        batch_user_s, batch_ment_s, batch_user_a, batch_ment_a, batch_r, \
            batch_user_next_s, batch_ment_next_s, batch_opposed_user_a, \
            batch_opposed_ment_a = self.buffer.sample(1024, device)

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
        self.soft_update(0.1, u_type='critic')
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

        user_log_probs = self.user_actor.get_log_probs(
            batch_user_s, batch_user_a[:, 1])
        ment_log_probs = self.ment_actor.get_log_probs(
            batch_ment_s, batch_ment_a[:, 1])
        loss_actor = -torch.mean(
            user_log_probs * user_A + ment_log_probs * ment_A)
        self.user_actor_optimizer.zero_grad()
        self.ment_actor_optimizer.zero_grad()
        loss_actor.backward()
        self.user_actor_optimizer.step()
        self.ment_actor_optimizer.step()
        self.soft_update(1, u_type='actor')

        return loss_critic.item(), loss_actor.item(), user_A.mean().item(), ment_A.mean().item()

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

    def joint_train(self, data, device):
        cf_loss, _, batch_acc, _ = self.update_buffer(
            data, device, need_backward=True,
            train_classifier=True, update_buffer=True)
        self.optimize_AC(device)
        return cf_loss, batch_acc

    def to(self, device):
        self.text_encoder.to(device)
        self.user_actor.to(device)
        self.ment_actor.to(device)
        self.target_user_actor.to(device)
        self.target_ment_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.classifier.to(device)

    def load_pre_trained(self, base):

        self.text_encoder.load_state_dict(
            torch.load(f"parameter/mixdan/text_encoder")
        )
        self.classifier.load_state_dict(
            torch.load(f"parameter/mixdan/classifier")
        )

    def classifier_benchmark(self, data, device):
        tweet, user_hs, ment_hs, ground_truth = \
            [item.to(device) for item in data]
        # user_hs: (batch, 33*50), batch size fixed to 1

        self.text_encoder.eval()
        self.classifier.eval()

        with torch.no_grad():
            feature = self.text_encoder(tweet, user_hs, ment_hs)
            # user_hs: (batch, 50, d_embed*2)
            if self.base == "dan":
                history = torch.cat(feature, dim=1)
                history = history.mean(dim=1)
                # (b, d_embed*2)

            output = self.classifier(history)
            output = output.squeeze(dim=-1)

            c_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
            batch_acc = output.ge(0).float() == ground_truth
            predict = output.ge(0).float()

        return c_loss.item(), output, batch_acc, ground_truth, predict
