import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import torch.optim as optim
import numpy as np


from models.classifier import DotAttention, SingleTextClassifier
from models.classifier import AddAttention
from models.embedNet import EmbedNet



torch.manual_seed(1)
torch.cuda.manual_seed(1)


class Buffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def append(self, user_s, ment_s, user_a, ment_a, r,
               user_next_s, ment_next_s, opposed_user_a, opposed_ment_a):

        transition = (user_s, ment_s, user_a, ment_a, r,
                      user_next_s, ment_next_s, opposed_user_a, opposed_ment_a)
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
        batch_r = torch.Tensor(batch_r).unsqueeze(-1).to(device)
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


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.att_encoder = DotAttention()

    def forward(self, tweet, history_a, history_b):
        _, encoding = self.att_encoder(
            tweet.view(-1, 1, 1, tweet.size()[1]), history_a, history_b)
        return encoding


class BaseActor(nn.Module):
    # TODO 不要依赖全局device
    def __init__(self, encoding_dim, device):
        super(BaseActor, self).__init__()
        self.encoding_dim = encoding_dim
        self.selected_data = [torch.zeros(1, 1, self.encoding_dim).to(device)]
        self.unselected_data = \
            [torch.zeros(1, 1, self.encoding_dim).to(device)]

    def forward(self, data, need_backward, is_target):
        raise NotImplementedError

    def choose_action(self, s, device, is_training, is_target):
        probs = self.map(s)
        if is_training:
            sampler = Categorical(probs)
            action = sampler.sample()
        else:
            action = torch.max(probs, 1, keepdim=False)[1]

        if is_target:
            actions = action.unsqueeze(1)
            return torch.zeros(s.size(0), 2).to(device).scatter_(1, actions, 1)
        else:
            return action.item()

    def get_log_probs(self, s, action):
        probs = self.map(s)
        sampler = Categorical(probs)
        log_probs = sampler.log_prob(action)
        return log_probs


class MLPActor(BaseActor):
    def __init__(self, d_embed, device):
        super(MLPActor, self).__init__(d_embed, device)
        self.map = nn.Sequential(
            nn.Linear(d_embed * 4, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 2),
            nn.Softmax(dim=-1))

    def forward(self, tweet, history, device, is_training, is_target):
        with torch.no_grad():
            state_a = torch.mean(
                torch.cat(self.selected_data, dim=1), dim=1, keepdim=True)
            state_b = torch.mean(
                torch.cat(self.unselected_data, dim=1), dim=1, keepdim=True)
            state = torch.cat([state_a, state_b, tweet, history], dim=-1).detach()
            state = state.squeeze(dim=1)
        # TODO 改with torch.no_grad()
        action = self.choose_action(state, device, is_training, is_target)
        # target actor cannot update the data buffer
        if not is_target:
            if action == 1:
                self.selected_data.append(history)
            else:
                self.unselected_data.append(history)
        return state, action


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


class MultiAgentClassifier():
    def __init__(self, n_vocab, d_embed, device):
        self.embedding = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)

        self.textEncoder = TextEncoder()
        self.user_actor = MLPActor(d_embed, device)
        self.ment_actor = MLPActor(d_embed, device)
        self.target_user_actor = MLPActor(d_embed, device)
        self.target_ment_actor = MLPActor(d_embed, device)
        self.critic = Critic(text_state_dim=d_embed * 4,
                             encoding_dim=d_embed)
        self.target_critic = Critic(text_state_dim=d_embed * 4,
                                    encoding_dim=d_embed)
        self.classifier = SingleTextClassifier(d_embed)

        self.load_pre_trained()
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
            [{"params": self.classifier.parameters()}])

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
        # user_hs: (batch, 33*50), batch size fixed to 1

        if not train_classifier:
            self.embedding = self.embedding.eval()
            self.user_embedding_a = self.user_embedding_a.eval()
            self.user_embedding_b = self.user_embedding_b.eval()
            self.ment_embedding_a = self.ment_embedding_a.eval()
            self.ment_embedding_b = self.ment_embedding_b.eval()

        tweet = self.embedding(tweet)
        user_hs_a = self.user_embedding_a(user_hs)
        user_hs_b = self.user_embedding_b(user_hs)
        ment_hs_a = self.ment_embedding_a(ment_hs)
        ment_hs_b = self.ment_embedding_b(ment_hs)
        # user_hs: (batch, 33*50, 200)

        user_hs_a, user_hs_b, ment_hs_a, ment_hs_b = map(
            lambda x: x.view(-1, 50, 33, 200),
            [user_hs_a, user_hs_b, ment_hs_a, ment_hs_b])
        tweet = torch.sum(tweet, dim=-2)
        user_hs = self.textEncoder(tweet, user_hs_a, user_hs_b)
        ment_hs = self.textEncoder(tweet, ment_hs_a, ment_hs_b)
        # user_hs: (batch, 50, 300)
        user_hs, ment_hs = map(
            lambda x: torch.chunk(x, 50, dim=1), [user_hs, ment_hs])

        tweet_num = len(user_hs)
        label = ground_truth.squeeze(0).item()
        # TODO is label useful?
        output = None

        tweet_t = tweet.unsqueeze(dim=1)
        select_index = list(np.random.randint(1, tweet_num, 3))
        for i in range(tweet_num-1):
            user_h, ment_h = user_hs[i], ment_hs[i]
            # user_h: (batch, 1, 200)
            next_user_h, next_ment_h = map(
                lambda x: x.detach(), [user_hs[i+1], ment_hs[i+1]])

            with torch.no_grad():
                user_s1, user_a1 = self.user_actor(tweet_t, user_h, device,
                                                   is_training=need_backward,
                                                   is_target=False)
                ment_s1, ment_a1 = self.ment_actor(tweet_t, ment_h, device,
                                                   is_training=need_backward,
                                                   is_target=False)

                if i in select_index and update_buffer is True:
                    selected_user_h = torch.cat(
                        self.user_actor.selected_data, dim=1)
                    selected_ment_h = torch.cat(
                        self.ment_actor.selected_data, dim=1)
                    output = self.classifier(
                        tweet, selected_ment_h, selected_user_h)

                    prob = torch.sigmoid(output).squeeze(dim=0).item()
                    score = label * prob + (1-label) * (1 - prob)
                    r = score - last_score

                    last_score = score
                    user_s2, _ = self.target_user_actor(tweet_t, next_user_h,
                                                        device,
                                                        is_training=False,
                                                        is_target=True)
                    ment_s2, _ = self.target_ment_actor(tweet_t, next_ment_h,
                                                        device,
                                                        is_training=False,
                                                        is_target=True)
                    user_a1 = [[1., 0.]] if user_a1 == 0 else [[0., 1.]]
                    opposed_user_a = [[user_a1[0][1], user_a1[0][0]]]
                    ment_a1 = [[1., 0.]] if ment_a1 == 0 else [[0., 1.]]
                    opposed_ment_a = [[ment_a1[0][1], ment_a1[0][0]]]
                    user_a1 = torch.Tensor(user_a1).to(device)
                    opposed_user_a = torch.Tensor(opposed_user_a).to(device)
                    ment_a1 = torch.Tensor(ment_a1).to(device)
                    opposed_ment_a = torch.Tensor(opposed_ment_a).to(device)
                    if update_buffer:
                        self.buffer.append(
                            user_s1, ment_s1, user_a1, ment_a1, r,
                            user_s2, ment_s2, opposed_user_a, opposed_ment_a)

        selected_user_h = torch.cat(self.user_actor.selected_data, dim=1)
        selected_ment_h = torch.cat(self.ment_actor.selected_data, dim=1)

        selected_num = (
            len(self.user_actor.selected_data) +
            len(self.ment_actor.selected_data)
            ) / 2

        output = self.classifier(tweet, selected_ment_h, selected_user_h)
        output = output.squeeze(dim=-1)
        c_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
        with torch.no_grad():
            batch_acc = output.ge(0).float() == ground_truth
            predict = output.ge(0).float()

        if train_classifier:
            self.classifier_optimizer.zero_grad()
            c_loss.backward()
            self.classifier_optimizer.step()
        self.reset_actor_selected_and_unselected_data(device)

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
            batch_user_next_a = self.target_user_actor.choose_action(
                batch_user_next_s, device, is_training=False, is_target=True)
            batch_ment_next_a = self.target_ment_actor.choose_action(
                batch_ment_next_s, device, is_training=False, is_target=True)

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
        loss_actor = -torch.sum(
            user_log_probs * user_A + ment_log_probs * ment_A)
        self.user_actor_optimizer.zero_grad()
        self.ment_actor_optimizer.zero_grad()
        loss_actor.backward()
        self.user_actor_optimizer.step()
        self.ment_actor_optimizer.step()
        self.soft_update(0.1, u_type='actor')

        return loss_critic.item(), loss_actor.item()

    def reset_actor_selected_and_unselected_data(self, device):
        self.user_actor.selected_data = \
            [torch.zeros(1, 1, self.user_actor.encoding_dim).to(device)]
        self.user_actor.unselected_data = \
            [torch.zeros(1, 1, self.user_actor.encoding_dim).to(device)]
        self.ment_actor.selected_data = \
            [torch.zeros(1, 1, self.ment_actor.encoding_dim).to(device)]
        self.ment_actor.unselected_data = \
            [torch.zeros(1, 1, self.ment_actor.encoding_dim).to(device)]

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
                target_param.data.copy_(target_param.data)
            for target_param, param in zip(self.target_ment_actor.parameters(),
                                           self.ment_actor.parameters()):
                target_param.data.copy_(target_param.data)
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
        tweet, user_h, ment_h, target = [item.to(device) for item in data]
        tweet, user_h, ment_h = map(embeddings, [tweet, user_h, ment_h])
        # user_h: (batch, 33*200, 300)

        user_h = user_h.view(-1, 200, 33, 300)
        ment_h = ment_h.view(-1, 200, 33, 300)

        output = self.classifier(tweet, user_h, ment_h)
        output.squeeze(dim=-1)
        c_loss = F.binary_cross_entropy_with_logits(output, target)

        self.classifier_optimizer.zero_grad()
        c_loss.backward()
        self.classifier_optimizer.step()
        return c_loss.item(), output

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

        self.embedding.embed.load_state_dict(
            torch.load("parameter/tweet_embed/embed"))
        self.user_embedding_a.embed.load_state_dict(
            torch.load("parameter/tweet_embed/embed_user_a"))
        self.user_embedding_b.embed.load_state_dict(
            torch.load("parameter/tweet_embed/embed_user_b"))
        self.ment_embedding_a.embed.load_state_dict(
            torch.load("parameter/tweet_embed/embed_ment_a"))
        self.ment_embedding_b.embed.load_state_dict(
            torch.load("parameter/tweet_embed/embed_ment_b"))

        self.classifier.block1.load_state_dict(
            {k: v for k, v in torch.load(
                "parameter/tweet_embed/block1").items()
                if k in self.classifier.block1.state_dict()})

        self.classifier.block2.load_state_dict(
            {k: v for k, v in torch.load(
                "parameter/tweet_embed/block2").items()
                if k in self.classifier.block2.state_dict()})

        self.classifier.block3.load_state_dict(
            {k: v for k, v in torch.load(
                "parameter/tweet_embed/block3").items()
                if k in self.classifier.block3.state_dict()})

        self.classifier.output_fc.load_state_dict(
            torch.load("parameter/tweet_embed/output_fc"))


class SimpleClassifier(nn.Module):
    def __init__(self, d_embed):
        super(SimpleClassifier, self).__init__()
        self.user_encoder = AddAttention(d_embed)
        self.ment_encoder = AddAttention(d_embed)
        self.output_fc = nn.Linear(d_embed, 1)
        nn.init.xavier_uniform_(self.output_fc.weight)

    def forward(self, tweet, user_h, ment_h):
        _, user_h = self.user_encoder(
            tweet.view(-1, 1, tweet.size()[1]), user_h)
        _, ment_h = self.ment_encoder(
            tweet.view(-1, 1, tweet.size()[1]), ment_h)

        merge = torch.stack(
            [user_h, ment_h, tweet], dim=0)
        hidden = torch.sum(merge, dim=0)

        logits = self.output_fc(hidden)
        return logits


class SMultiAgentClassifier():
    def __init__(self, n_vocab, d_embed, device):
        self.embedding = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_actor = MLPActor(d_embed, device)
        self.ment_actor = MLPActor(d_embed, device)
        self.target_user_actor = MLPActor(d_embed, device)
        self.target_ment_actor = MLPActor(d_embed, device)
        self.critic = Critic(text_state_dim=d_embed * 4,
                             encoding_dim=d_embed)
        self.target_critic = Critic(text_state_dim=d_embed * 4,
                                    encoding_dim=d_embed)
        self.classifier = SimpleClassifier(d_embed)

        self.embedding.to(device)
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
             {"params": self.embedding.parameters()}], weight_decay=1e-8)

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
        # user_hs: (batch, 33*200), batch size fixed to 1

        if not train_classifier:
            self.embedding = self.embedding.eval()

        # tweet, user_hs, ment_hs = map(
        #     self.embedding, [tweet, user_hs, ment_hs])
        # user_hs: (batch, 33*200, 300)
        user_hs = torch.rand(1, 200, 300).to(device)
        ment_hs = torch.rand(1, 200, 300).to(device)
        tweet = torch.rand(1, 300).to(device)
        # user_hs, ment_hs = map(
        #     lambda x: x.view(-1, 200, 33, 300), [user_hs, ment_hs])
        # tweet_t = torch.sum(tweet, dim=-2)
        # user_hs = self.textEncoder(tweet_t, user_hs)
        # ment_hs = self.textEncoder(tweet_t, ment_hs)
        # user_hs: (batch, 200, 300)
        tweet_t = tweet
        # user_hs, ment_hs = map(
        #     lambda x: torch.chunk(x, 50, dim=1), [user_hs, ment_hs])

        tweet_num = len(user_hs)
        label = ground_truth.squeeze(0).item()
        # TODO is label useful?
        output = None
        # with torch.no_grad():
        #     tweet_t = tweet_t.unsqueeze(dim=1)
        #     for i in range(tweet_num-1):
        #         user_h, ment_h = user_hs[i], ment_hs[i]
        #         # user_h: (batch, 1, 300)
        #         next_user_h, next_ment_h = map(
        #             lambda x: x.detach(), [user_hs[i+1], ment_hs[i+1]])

        #         user_s1, user_a1 = self.user_actor(tweet_t, user_h, device,
        #                                            is_training=need_backward,
        #                                            is_target=False)
        #         ment_s1, ment_a1 = self.ment_actor(tweet_t, ment_h, device,
        #                                            is_training=need_backward,
        #                                            is_target=False)

        #         selected_user_h = torch.cat(
        #             self.user_actor.selected_data, dim=1)
        #         selected_ment_h = torch.cat(
        #             self.ment_actor.selected_data, dim=1)
        #         output = self.classifier(
        #             tweet, selected_ment_h, selected_user_h)

        #         prob = torch.sigmoid(output).squeeze(dim=0).item()
        #         score = label * prob + (1-label) * (1 - prob)
        #         r = score - last_score

        #         last_score = score
        #         user_s2, _ = self.target_user_actor(tweet_t, next_user_h,
        #                                             device,
        #                                             is_training=False,
        #                                             is_target=True)
        #         ment_s2, _ = self.target_ment_actor(tweet_t, next_ment_h,
        #                                             device,
        #                                             is_training=False,
        #                                             is_target=True)
        #         user_a1 = [[1., 0.]] if user_a1 == 0 else [[0., 1.]]
        #         opposed_user_a = [[user_a1[0][1], user_a1[0][0]]]
        #         ment_a1 = [[1., 0.]] if ment_a1 == 0 else [[0., 1.]]
        #         opposed_ment_a = [[ment_a1[0][1], ment_a1[0][0]]]
        #         user_a1 = torch.Tensor(user_a1).to(device)
        #         opposed_user_a = torch.Tensor(opposed_user_a).to(device)
        #         ment_a1 = torch.Tensor(ment_a1).to(device)
        #         opposed_ment_a = torch.Tensor(opposed_ment_a).to(device)
        #         if update_buffer:
        #             self.buffer.append(
        #                 user_s1, ment_s1, user_a1, ment_a1, r,
        #                 user_s2, ment_s2, opposed_user_a, opposed_ment_a)

        # selected_user_h = torch.cat(self.user_actor.selected_data, dim=1)
        # selected_ment_h = torch.cat(self.ment_actor.selected_data, dim=1)
        selected_user_h = user_hs
        selected_ment_h = ment_hs
        output = self.classifier(tweet, selected_ment_h, selected_user_h)
        output = output.squeeze(dim=-1)
        c_loss = F.binary_cross_entropy_with_logits(output, ground_truth)
        with torch.no_grad():
            batch_acc = output.ge(0).float() == ground_truth

        if train_classifier:
            self.classifier_optimizer.zero_grad()
            c_loss.backward()
            self.classifier_optimizer.step()
        self.reset_actor_selected_and_unselected_data(device)

        return c_loss.item(), output.detach(), batch_acc, ground_truth

    def optimize_AC(self, device):
        # update critic
        return
        batch_user_s, batch_ment_s, batch_user_a, batch_ment_a, batch_r, \
            batch_user_next_s, batch_ment_next_s, batch_opposed_user_a, \
            batch_opposed_ment_a = self.buffer.sample(1024, device)

        Q_predicted = self.critic(
            batch_user_s, batch_user_a, batch_ment_s, batch_ment_a)

        with torch.no_grad():
            batch_user_next_a = self.target_user_actor.choose_action(
                batch_user_next_s, device, is_training=False, is_target=True)
            batch_ment_next_a = self.target_ment_actor.choose_action(
                batch_ment_next_s, device, is_training=False, is_target=True)

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
        loss_actor = -torch.sum(
            user_log_probs * user_A + ment_log_probs * ment_A)
        self.user_actor_optimizer.zero_grad()
        self.ment_actor_optimizer.zero_grad()
        loss_actor.backward()
        self.user_actor_optimizer.step()
        self.ment_actor_optimizer.step()
        self.soft_update(0.1, u_type='actor')

    def reset_actor_selected_and_unselected_data(self, device):
        self.user_actor.selected_data = \
            [torch.zeros(1, 1, self.user_actor.encoding_dim).to(device)]
        self.user_actor.unselected_data = \
            [torch.zeros(1, 1, self.user_actor.encoding_dim).to(device)]
        self.ment_actor.selected_data = \
            [torch.zeros(1, 1, self.ment_actor.encoding_dim).to(device)]
        self.ment_actor.unselected_data = \
            [torch.zeros(1, 1, self.ment_actor.encoding_dim).to(device)]

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
                target_param.data.copy_(target_param.data)
            for target_param, param in zip(self.target_ment_actor.parameters(),
                                           self.ment_actor.parameters()):
                target_param.data.copy_(target_param.data)
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
        tweet, user_h, ment_h, target = [item.to(device) for item in data]
        tweet, user_h, ment_h = map(embeddings, [tweet, user_h, ment_h])
        # user_h: (batch, 33*200, 300)

        user_h = user_h.view(-1, 200, 33, 300)
        ment_h = ment_h.view(-1, 200, 33, 300)

        output = self.classifier(tweet, user_h, ment_h)
        output.squeeze(dim=-1)
        c_loss = F.binary_cross_entropy_with_logits(output, target)

        self.classifier_optimizer.zero_grad()
        c_loss.backward()
        self.classifier_optimizer.step()
        return c_loss.item(), output

    def joint_train(self, data, device):
        cf_loss, _, batch_acc, _ = self.update_buffer(
            data, device, need_backward=True,
            train_classifier=True, update_buffer=True)
        self.optimize_AC(device)
        return cf_loss, batch_acc
