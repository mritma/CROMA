import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.transformer import EncoderLayer
from models.embedNet import EmbedNet


class DotAttention(nn.Module):

    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, query, context, value=None,
                context_mask=None, scale=1):
        q, c, v = query, context, value
        if v is None:
            v = c

        score = torch.mul(q, c).sum(dim=-1)
        score = torch.div(score, scale)
        if context_mask is not None:
            score = score.masked_fill(context_mask, -1e25)
        weight = F.softmax(score, dim=-1).unsqueeze(dim=-1)

        z = torch.mul(weight, v).sum(dim=-2)

        return weight, z


class AddAttention(nn.Module):

    def __init__(self, dim):
        super(AddAttention, self).__init__()
        self.map_query = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.map_query.weight)
        self.map_context = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.map_context.weight)
        self.map_score = nn.Linear(dim, 1, bias=False)
        nn.init.xavier_uniform_(self.map_score.weight)

    def forward(self, query, context, value=None, context_mask=None):
        q, c, v = query, context, value
        # q: (b, 1, 200), c: (b, 50, 200), v: (b, 50, 200), m: (b, 50)
        if v is None:
            v = c

        score = self.map_score(
            torch.tanh(torch.add(self.map_query(q), self.map_context(c))))

        score = score.squeeze(dim=-1)
        # score: (b, 50)
        if context_mask is not None:
            context_mask = 1 - context_mask
            context_mask = context_mask.byte()
            score = score.masked_fill(context_mask, -1e25)
        weight = F.softmax(score, dim=-1).unsqueeze(dim=-1)
        if context_mask is not None and context_mask.float().mean().item() == 1:
            weight = weight * 0
        # weight: (b, 50, 1)
        z = torch.mul(weight, v).sum(dim=-2)
        # z: (b, 200)

        return weight, z


class AttentionBlock(nn.Module):

    def __init__(self, d_embed):
        super(AttentionBlock, self).__init__()

        self.user_word_attn = DotAttention()
        self.ment_word_attn = DotAttention()
        self.user_sent_attn = AddAttention(d_embed)
        self.ment_sent_attn = AddAttention(d_embed)

    def forward(self, hidden, history_a, history_b=None):
        if history_b is None:
            history_b = history_a

        user_x_a = history_a[0]
        ment_x_a = history_a[1]

        user_x_b = history_b[0]
        ment_x_b = history_b[1]

        _, user_history = self.user_word_attn(
            hidden.view(-1, 1, 1, hidden.size()[1]), user_x_a, user_x_b)

        _, user_history = self.user_sent_attn(
            hidden.view(-1, 1, hidden.size()[1]), user_history)

        _, ment_history = self.ment_word_attn(
            hidden.view(-1, 1, 1, hidden.size()[1]), ment_x_a, ment_x_b)

        _, ment_history = self.ment_sent_attn(
            hidden.view(-1, 1, hidden.size()[1]), ment_history)

        merge = torch.stack(
            [user_history, ment_history, hidden], dim=0)
        hidden = torch.sum(merge, dim=0)

        return hidden


class SentAttentionBlock(nn.Module):

    def __init__(self, d_embed):
        super(SentAttentionBlock, self).__init__()

        self.tweet_fc = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(inplace=True),
            nn.Linear(d_embed, 300)
        )
        self.user_fc = nn.Sequential(
            nn.Linear(d_embed*2, d_embed),
            nn.ReLU(inplace=True),
            nn.Linear(d_embed, 300)
        )
        self.ment_fc = nn.Sequential(
            nn.Linear(d_embed*2, d_embed),
            nn.ReLU(inplace=True),
            nn.Linear(d_embed, 300)
        )

        self.user_sent_attn = AddAttention(300)
        self.ment_sent_attn = AddAttention(300)
        self.merge = nn.Linear(300*3, d_embed)

    def forward(self, tweet, user_h, ment_h):

        tmp_t = torch.stack([tweet]*user_h.size(1), dim=1)
        tweet = self.tweet_fc(tweet)
        user_h = torch.cat([user_h, tmp_t], dim=-1)
        ment_h = torch.cat([ment_h, tmp_t], dim=-1)
        user_h = self.user_fc(user_h)
        ment_h = self.ment_fc(ment_h)
        del tmp_t

        _, user_h = self.user_sent_attn(
            tweet.view(-1, 1, tweet.size()[1]), user_h)

        _, ment_h = self.ment_sent_attn(
            tweet.view(-1, 1, tweet.size()[1]), ment_h)

        # merge = torch.stack(
        #     [user_h, ment_h, tweet], dim=0)
        # hidden = torch.sum(merge, dim=0)
        hidden = self.merge(torch.cat([user_h, ment_h, tweet], dim=-1))

        return hidden


class SentClassifier(nn.Module):
    """User sent feature to classify"""
    def __init__(self, d_embed, dropout=0.2):
        super(SentClassifier, self).__init__()
        self.p = dropout
        self.user_encoder = AddAttention(d_embed)
        self.ment_encoder = AddAttention(d_embed)
        self.output_fc = nn.Linear(d_embed, 1)
        nn.init.xavier_uniform_(self.output_fc.weight)

    def forward(self, tweet, user_h, ment_h):
        tweet = F.dropout(tweet, p=self.p)
        user_h = F.dropout(user_h, p=self.p)
        ment_h = F.dropout(ment_h, p=self.p)
        _, user_h = self.user_encoder(
            tweet.view(-1, 1, tweet.size()[1]), user_h)
        _, ment_h = self.ment_encoder(
            tweet.view(-1, 1, tweet.size()[1]), ment_h)

        merge = torch.stack(
            [user_h, ment_h, tweet], dim=0)
        hidden = torch.sum(merge, dim=0)
        hidden = torch.tanh(hidden)

        logits = self.output_fc(hidden)
        return logits


class WordClassifier(nn.Module):
    """User word feature to classify"""
    def __init__(self, d_embed, dropout=0.2, freeze=False):
        super(WordClassifier, self).__init__()
        self.p = dropout
        self.freeze = freeze
        self.embedding = nn.Embedding(50001, d_embed)
        self.user_encoder = AddAttention(d_embed)
        self.ment_encoder = AddAttention(d_embed)
        self.output_fc = nn.Linear(d_embed, 1)
        nn.init.xavier_uniform_(self.output_fc.weight)

        with open("Dataset/tweet_embed_tensor.pkl", 'rb') as f:
            self.embedding.from_pretrained(torch.load(f), freeze=self.freeze)

    def forward(self, tweet, user_h, ment_h):
        tweet = self.embedding(tweet)
        user_h = self.embedding(user_h)
        ment_h = self.embedding(ment_h)
        tweet = F.dropout(tweet, p=self.p)
        user_h = F.dropout(user_h, p=self.p)
        ment_h = F.dropout(ment_h, p=self.p)

        tweet = torch.mean(tweet, dim=-2)
        user_h = torch.mean(user_h, dim=-2)
        ment_h = torch.mean(ment_h, dim=-2)

        _, user_h = self.user_encoder(
            tweet.view(-1, 1, tweet.size()[1]), user_h)
        _, ment_h = self.ment_encoder(
            tweet.view(-1, 1, tweet.size()[1]), ment_h)

        merge = torch.stack(
            [user_h, ment_h, tweet], dim=0)
        hidden = torch.sum(merge, dim=0)
        hidden = torch.tanh(hidden)

        logits = self.output_fc(hidden)
        return logits


class SingleAttentionBlock(nn.Module):

    def __init__(self, d_embed):
        super(SingleAttentionBlock, self).__init__()

        self.user_sent_attn = AddAttention(d_embed)
        self.ment_sent_attn = AddAttention(d_embed)

    def forward(self, hidden, history_a, history_b=None, mask=None):

        user_history = history_a[0]
        ment_history = history_a[1]

        user_mask = mask[0]
        ment_mask = mask[1]

        user_weight, user_history = self.user_sent_attn(
            hidden.view(-1, 1, hidden.size()[1]),
            user_history, context_mask=user_mask
        )

        ment_weight, ment_history = self.ment_sent_attn(
            hidden.view(-1, 1, hidden.size()[1]),
            ment_history, context_mask=ment_mask
        )

        merge = torch.stack(
            [user_history, ment_history, hidden], dim=0)

        hidden = torch.sum(merge, dim=0)

        return hidden, (user_weight, ment_weight)


class SingleTextClassifier(nn.Module):

    def __init__(self, d_embed):
        super(SingleTextClassifier, self).__init__()

        self.block1 = SingleAttentionBlock(d_embed)
        self.block2 = SingleAttentionBlock(d_embed)
        self.block3 = SingleAttentionBlock(d_embed)

        self.output_fc = nn.Linear(d_embed, 1)
        nn.init.xavier_uniform_(self.output_fc.weight)

    def forward(self, tweet, user_h, ment_h, user_mask=None, ment_mask=None):
        # t: (b, d_embed), h: (b, 50, 200), m: (b, 50)
        hidden, _ = self.block1(
            tweet, (user_h, ment_h), mask=(user_mask, ment_mask)
        )
        hidden, _ = self.block2(
            hidden, (user_h, ment_h), mask=(user_mask, ment_mask)
        )
        hidden, attn_weight = self.block3(
            hidden, (user_h, ment_h), mask=(user_mask, ment_mask)
        )

        logits = self.output_fc(hidden)

        return logits, hidden, attn_weight


class NoIterClassifier(nn.Module):

    def __init__(self, d_embed):
        super(NoIterClassifier, self).__init__()

        self.block1 = SingleAttentionBlock(d_embed)

        self.output_fc = nn.Linear(d_embed, 1)
        nn.init.xavier_uniform_(self.output_fc.weight)

    def forward(self, tweet, user_h, ment_h, user_mask=None, ment_mask=None):
        # t: (b, d_embed), h: (b, 50, 200), m: (b, 50)
        hidden = self.block1(
            tweet, (user_h, ment_h), mask=(user_mask, ment_mask)
        )

        logits = self.output_fc(hidden)

        return logits, hidden


class TransformerClassifier(nn.Module):

    def __init__(self, d_embed, heads):
        super(TransformerClassifier, self).__init__()

        self.attn = EncoderLayer(d_embed, heads)
        self.output_fc = nn.Linear(d_embed, 1)

    def forward(self, tweet, user_h, ment_h, user_mask=None, ment_mast=None):
        # t: (b, d_embed), h: (b, 50, 200), m: (b, 50)
        t = tweet.view(tweet.size(0), 1, tweet.size(1))
        x = torch.cat([t, user_h, ment_h], dim=1)
        x = self.attn(x)
        # x: (b, 101, 200)
        x = x.mean(dim=1)
        logits = self.output_fc(x)

        return logits


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.att_encoder = DotAttention()

    def forward(self, tweet, history_a, history_b):
        _, encoding = self.att_encoder(
            tweet.view(-1, 1, 1, tweet.size()[1]), history_a, history_b)
        return encoding


class WordModel(nn.Module):
    def __init__(self, n_vocab, d_embed, c_name, heads=None):
        super(WordModel, self).__init__()

        self.embedding = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)

        self.text_encoder = TextEncoder()
        if c_name == "transformer":
            assert heads is not None, "How many heads for transformer?"
            self.classifier = TransformerClassifier(d_embed, heads)
        if c_name == "origin":
            self.classifier = SingleTextClassifier(d_embed)
        if c_name == "no_iter":
            self.classifier = NoIterClassifier(d_embed)

    def forward(self, tweet, user_hs, ment_hs, u_mask=None, m_mask=None):
        tweet = self.embedding(tweet)
        user_hs_a = self.user_embedding_a(user_hs)
        user_hs_b = self.user_embedding_b(user_hs)
        ment_hs_a = self.ment_embedding_a(ment_hs)
        ment_hs_b = self.ment_embedding_b(ment_hs)

        # user_hs_a, user_hs_b, ment_hs_a, ment_hs_b = map(
        #     lambda x: x.view(-1, 50, 33, 200),
        #     [user_hs_a, user_hs_b, ment_hs_a, ment_hs_b])
        tweet = torch.sum(tweet, dim=-2)
        user_hs = self.text_encoder(tweet, user_hs_a, user_hs_b)
        ment_hs = self.text_encoder(tweet, ment_hs_a, ment_hs_b)
        # user_hs: (batch, 50, 200)

        output, _, _ = self.classifier(tweet, user_hs, ment_hs, u_mask, m_mask)

        return output


class DAN(nn.Module):
    """对DAN，问题可以建模成tweet和history（user+mention混合）是否match，
       则可以用m-DAN， 在tweet和history间做dual-attn
       同一个模块调用多次，会不会梯度爆炸？怎么解决？或者encoder用attention？
    """
    def __init__(self, n_vocab, embedding_size):
        super(DAN, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.textEncoder = nn.LSTM(input_size=embedding_size,
                                   hidden_size=embedding_size//2,
                                   bidirectional=True,
                                   num_layers=1,
                                   batch_first=True)
        # parameters for tweet Attention.
        self.W_t = nn.Linear(embedding_size, embedding_size)
        self.W_tm = nn.Linear(embedding_size, embedding_size)
        self.W_th = nn.Linear(embedding_size, 1)
        # parameters for history Attention.
        self.W_h = nn.Linear(embedding_size, embedding_size)
        self.W_hm = nn.Linear(embedding_size, embedding_size)
        self.W_hh = nn.Linear(embedding_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

    def forward(self, tweet, user_hs, ment_hs):
        with torch.no_grad():
            tweet = tweet.unsqueeze(dim=1)
            histories = torch.cat([tweet, user_hs, ment_hs], dim=-2)
            # histories (b, 101, 33)
        histories = self.embedding(histories)
        # (b, 101, 33, d_embed)
        histories = torch.split(histories, 1, dim=1)
        histories = [h.squeeze(dim=1) for h in histories]
        # h (b, 33, d_embed)
        tweet = histories[0]
        tweet, (_, _) = self.textEncoder(tweet)
        # tweet (b, 33, d_embed)

        m = [self.dual_att(tweet, h) for h in histories[1:]]
        # list of (b, d_embed*2)
        m = torch.mean(torch.stack(m, dim=1), dim=1)
        # m: (b, d_embed*2)
        logits = self.classifier(m)
        return logits

    def dual_att(self, tweet, history):
        history, (_, _) = self.textEncoder(history)
        # h (b, 33, d_embed)
        t0, h0 = map(lambda x: x.mean(dim=1, keepdim=True), [tweet, history])

        # h-attn:
        h_h = torch.tanh(self.W_h(history)) * torch.tanh(self.W_hm(t0))
        # h_h (b, 33, d_embed)
        alpha_h = F.softmax(self.W_hh(h_h), dim=1)
        # alpha_h (b, 33, 1)
        h = torch.mul(history, alpha_h).sum(dim=1)
        # h: (b, d_embed)

        # t-attn
        h_t = torch.tanh(self.W_t(tweet)) * torch.tanh(self.W_tm(h0))
        alpha_t = F.softmax(self.W_th(h_t), dim=1)
        t = torch.mul(tweet, alpha_t).sum(dim=1)
        # t: (b, d_embed)

        return torch.cat([t, h], dim=-1)


class CAN(nn.Module):
    def __init__(self, n_vocab, embedding_size):
        super(CAN, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.textEncoder = nn.LSTM(input_size=embedding_size,
                                   hidden_size=embedding_size//2,
                                   bidirectional=True,
                                   num_layers=1,
                                   batch_first=True)
        self.W_xh = nn.Linear(embedding_size, embedding_size)
        self.W_gh = nn.Linear(embedding_size, embedding_size)
        self.W_ph = nn.Linear(embedding_size, 1)

        self.W_xt = nn.Linear(embedding_size, embedding_size)
        self.W_gt = nn.Linear(embedding_size, embedding_size)
        self.W_pt = nn.Linear(embedding_size, 1)

        self.W_f = nn.Linear(embedding_size, 1)

    def forward(self, tweet, user_hs, ment_hs):
        with torch.no_grad():
            tweet = tweet.unsqueeze(dim=1)
            histories = torch.cat([tweet, user_hs, ment_hs], dim=-2)
            # histories (b, 101, 33)
        histories = self.embedding(histories)
        # (b, 101, 33, d_embed)
        histories = torch.split(histories, 1, dim=1)
        histories = [h.squeeze(dim=1) for h in histories]
        # h (b, 33, d_embed)
        tweet = histories[0]
        tweet, (_, _) = self.textEncoder(tweet)
        # tweet (b, 33, d_embed)

        features = []
        for history in histories[1:]:
            h, (_, _) = self.textEncoder(history)
            # h (b, 33, d_embed)
            h = self.tweet_guided_att(history, tweet)
            t = self.history_guided_att(tweet, h)
            f = h + t
            features.append(f)
        feature = torch.mean(torch.stack(features, dim=1), dim=1)
        # feature (b, d_embed)
        logits = self.W_f(feature)
        return logits

    def tweet_guided_att(self, history, tweet):
        t = torch.mean(tweet, dim=1, keepdim=True)
        # t: (b, 1, d_embed)
        h_h = torch.tanh(self.W_xh(history) + self.W_gh(t))
        # h_h: (b, 33, d_embed)
        alpha_h = F.softmax(self.W_ph(h_h), dim=1)
        # alpha_h: (b, 33, 1)
        h = torch.mul(history, alpha_h).sum(dim=1)
        # h: (b, d_embed)
        return h

    def history_guided_att(self, tweet, h):
        h = h.unsqueeze(dim=1)
        t_h = torch.tanh(self.W_xt(tweet) + self.W_gt(h))
        alpha_t = F.softmax(self.W_pt(t_h), dim=1)
        t = torch.mul(tweet, alpha_t).sum(dim=1)
        return t


class MAN(nn.Module):
    def __init__(self, n_vocab, embedding_size):
        super(MAN, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.textEncoder = nn.LSTM(input_size=embedding_size,
                                   hidden_size=embedding_size//2,
                                   bidirectional=True,
                                   num_layers=1,
                                   batch_first=True)
        self.W_attn = nn.Linear(embedding_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

    def forward(self, tweet, user_hs, ment_hs):
        with torch.no_grad():
            tweet = tweet.unsqueeze(dim=1)
            histories = torch.cat([tweet, user_hs, ment_hs], dim=-2)
            # histories (b, 101, 33)
        histories = self.embedding(histories)
        # (b, 101, 33, d_embed)
        histories = torch.split(histories, 1, dim=1)
        histories = [h.squeeze(dim=1) for h in histories]
        # h (b, 33, d_embed)
        tweet = histories[0]
        _, (tweet, _) = self.textEncoder(tweet)
        # tweet (b, 2, d_embed//2)
        tweet = tweet.transpose(0, 1)
        tweet = tweet.contiguous().view(tweet.size(0), 1, -1)

        features = []
        for history in histories[1:]:
            _, (h, _) = self.textEncoder(history)
            h = h.transpose(0, 1)
            h = h.contiguous().view(h.size(0), 1, -1)
            # h: (b, 1, d_embed)
            alpha_t = self.W_attn(tweet)
            alpha_h = self.W_attn(h)
            # alpha_x: (b, 1, 1)
            alpha = F.softmax(torch.cat([alpha_t, alpha_h], dim=1), dim=1)
            # alpha: (b, 2, 1)
            feature = torch.mul(alpha, torch.cat([tweet, h], dim=1)).sum(dim=1)
            # feature: (b, d_embed)
            features.append(feature)
        mean_feature = torch.mean(torch.stack(features, dim=1), dim=1)
        logits = self.classifier(mean_feature)

        return logits


class DANEncoder(nn.Module):
    def __init__(self, n_vocab, embedding_size):
        super(DANEncoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.textEncoder = nn.LSTM(input_size=embedding_size,
                                   hidden_size=embedding_size//2,
                                   bidirectional=True,
                                   num_layers=1,
                                   batch_first=True)
        # parameters for tweet Attention.
        self.W_t = nn.Linear(embedding_size, embedding_size)
        self.W_tm = nn.Linear(embedding_size, embedding_size)
        self.W_th = nn.Linear(embedding_size, 1)
        # parameters for history Attention.
        self.W_h = nn.Linear(embedding_size, embedding_size)
        self.W_hm = nn.Linear(embedding_size, embedding_size)
        self.W_hh = nn.Linear(embedding_size, 1)

    def forward(self, tweet, user_hs, ment_hs):
        with torch.no_grad():
            tweet = tweet.unsqueeze(dim=1)
            histories = torch.cat([tweet, user_hs, ment_hs], dim=-2)
            # histories (b, 101, 33)
        histories = self.embedding(histories)
        # (b, 101, 33, d_embed)
        histories = torch.split(histories, 1, dim=1)
        histories = [h.squeeze(dim=1) for h in histories]
        # h (b, 33, d_embed)
        tweet = histories[0]
        tweet, (_, _) = self.textEncoder(tweet)
        # tweet (b, 33, d_embed)

        m = [self.dual_att(tweet, h) for h in histories[1:]]
        # list of (b, d_embed*2)
        m = torch.stack(m, dim=1)
        # m: (b, 100, d_embed*2)
        user_hs, ment_hs = torch.chunk(m, 2, dim=1)
        return user_hs, ment_hs

    def dual_att(self, tweet, history):
        history, (_, _) = self.textEncoder(history)
        # h (b, 33, d_embed)
        t0, h0 = map(lambda x: x.mean(dim=1, keepdim=True), [tweet, history])

        # h-attn:
        h_h = torch.tanh(self.W_h(history)) * torch.tanh(self.W_hm(t0))
        # h_h (b, 33, d_embed)
        alpha_h = F.softmax(self.W_hh(h_h), dim=1)
        # alpha_h (b, 33, 1)
        h = torch.mul(history, alpha_h).sum(dim=1)
        # h: (b, d_embed)

        # t-attn
        h_t = torch.tanh(self.W_t(tweet)) * torch.tanh(self.W_tm(h0))
        alpha_t = F.softmax(self.W_th(h_t), dim=1)
        t = torch.mul(tweet, alpha_t).sum(dim=1)
        # t: (b, d_embed)

        return torch.cat([t, h], dim=-1)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, output_size)
        )

    def forward(self, input):
        output = self.classifier(input)

        return output


class DANClassifier(nn.Module):
    def __init__(self, n_vocab, d_embed):
        super(DANClassifier, self).__init__()
        self.encoder = DANEncoder(n_vocab, d_embed)
        self.classifier = MLPClassifier(d_embed*2, 1)

    def forward(self, tweet, user_hs, ment_hs):
        user_hs, ment_hs = self.encoder(tweet, user_hs, ment_hs)
        history = torch.mean(torch.cat([user_hs, ment_hs], dim=1), dim=1)
        logits = self.classifier(history)

        return logits


class GRUEncoder(nn.Module):
    def __init__(self, n_vocab, d_embed):
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, d_embed)
        self.text_encoder = nn.GRU(input_size=d_embed,
                                   hidden_size=d_embed//2,
                                   bidirectional=True,
                                   num_layers=1,
                                   batch_first=True)

    def forward(self, tweet, user_hs, ment_hs):
        with torch.no_grad():
            h_len = user_hs.size(1)
            tweet = tweet.unsqueeze(dim=1)
            histories = torch.cat([tweet, user_hs, ment_hs], dim=-2)
            # histories (b, 101, 33)
        histories = self.embedding(histories)
        # (b, 101, 33, d_embed)
        histories = torch.split(histories, 1, dim=1)
        histories = [h.squeeze(dim=1) for h in histories]
        # h (b, 33, d_embed)
        tweet = histories[0]
        user_hs = histories[1: h_len+1]
        ment_hs = histories[h_len+1: 101]
        tweet, (_, _) = self.text_encoder(tweet)
        # tweet (b, 33, d_embed)
        tweet = tweet.mean(dim=1)
        # (b, d_embed)

        m = [self.text_encoder(h)[0].mean(dim=1) for h in histories[1:]]
        # list of (b, d_embed)
        user_hs = torch.stack(m[:h_len], dim=1).mean(dim=1)
        ment_hs = torch.stack(m[h_len:], dim=1).mean(dim=1)
        feature = torch.cat([tweet, user_hs, ment_hs], dim=-1)
        # (b, d_embed*3)
        return feature


class GRUClassifier(nn.Module):
    def __init__(self, n_vocab, d_embed):
        super(GRUClassifier, self).__init__()
        self.encoder = GRUEncoder(n_vocab, d_embed)
        self.classifier = MLPClassifier(d_embed*3, 1)

    def forward(self, tweet, user_hs, ment_hs):
        feature = self.encoder(tweet, user_hs, ment_hs)
        logits = self.classifier(feature)

        return logits


class MixDANEncoder(nn.Module):
    def __init__(self, n_vocab, d_embed):
        super(MixDANEncoder, self).__init__()
        self.embedding = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.user_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_a = EmbedNet(n_vocab, d_embed, drop=0.2)
        self.ment_embedding_b = EmbedNet(n_vocab, d_embed, drop=0.2)

        self.W_t = nn.Linear(d_embed, d_embed)
        self.W_tm = nn.Linear(d_embed, d_embed)
        self.W_th = nn.Linear(d_embed, 1)
        # parameters for history Attention.
        self.W_h = nn.Linear(d_embed, d_embed)
        self.W_hm = nn.Linear(d_embed, d_embed)
        self.W_hh = nn.Linear(d_embed, 1)

    def forward(self, tweet, user_hs, ment_hs):
        tweet = self.embedding(tweet)
        # tweet: (b, 33, 200)
        user_hs_a = self.user_embedding_a(user_hs)
        user_hs_b = self.user_embedding_b(user_hs)
        ment_hs_a = self.ment_embedding_a(ment_hs)
        ment_hs_b = self.ment_embedding_b(ment_hs)
        # h: (b, 50, 33, 200)

        hs_a = torch.cat([user_hs_a, ment_hs_a], dim=1)
        hs_b = torch.cat([user_hs_b, ment_hs_b], dim=1)
        # h: (b, 100, 33, 200)
        hs_a = torch.split(hs_a, 1, dim=1)
        hs_b = torch.split(hs_b, 1, dim=1)
        # h: (b, 1, 33, 200)
        hs_a = [h.squeeze(dim=1) for h in hs_a]
        hs_b = [h.squeeze(dim=1) for h in hs_b]
        # h: (b, 33, 200)

        m = [self.dual_att(tweet, hs_a[i], hs_b[i]) for i in range(len(hs_a))]
        m = torch.stack(m, dim=1)
        # m: (b, 100, 200*2)
        user_hs, ment_hs = torch.chunk(m, 2, dim=1)
        return user_hs, ment_hs

    def dual_att(self, tweet, h_a, h_b):
        t0, h0 = map(lambda x: x.mean(dim=1, keepdim=True), [tweet, h_a])

        # h-attn:
        h_h = torch.tanh(self.W_h(h_a)) * torch.tanh(self.W_hm(t0))
        # h_h (b, 33, d_embed)
        alpha_h = F.softmax(self.W_hh(h_h), dim=1)
        # alpha_h (b, 33, 1)
        h = torch.mul(h_b, alpha_h).sum(dim=1)
        # h: (b, d_embed)

        # t-attn
        h_t = torch.tanh(self.W_t(tweet)) * torch.tanh(self.W_tm(h0))
        alpha_t = F.softmax(self.W_th(h_t), dim=1)
        t = torch.mul(tweet, alpha_t).sum(dim=1)
        # t: (b, d_embed)

        return torch.cat([t, h], dim=-1)


class MixDAN(nn.Module):
    def __init__(self, n_vocab, d_embed):
        super(MixDAN, self).__init__()
        self.text_encoder = MixDANEncoder(n_vocab, d_embed)
        self.classifier = MLPClassifier(d_embed*2, 1)

    def forward(self, tweet, user_hs, ment_hs):
        user_hs, ment_hs = self.text_encoder(tweet, user_hs, ment_hs)
        history = torch.mean(torch.cat([user_hs, ment_hs], dim=1), dim=1)
        logits = self.classifier(history)

        return logits


class LSTMEncoder(nn.Module):
    def __init__(self, n_vocab, d_embed):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, d_embed)
        self.text_encoder = nn.LSTM(
            input_size=d_embed, hidden_size=d_embed//2,
            bidirectional=True, num_layers=1,
            batch_first=True
        )

    def forward(self, tweet, user_hs, ment_hs):
        with torch.no_grad():
            h_len = user_hs.size(1)
            tweet = tweet.unsqueeze(dim=1)
            histories = torch.cat([tweet, user_hs, ment_hs], dim=-2)
            # histories (b, 101, 33)
        histories = self.embedding(histories)
        # (b, 101, 33, d_embed)
        histories = torch.split(histories, 1, dim=1)
        histories = [h.squeeze(dim=1) for h in histories]
        # h (b, 33, d_embed)
        tweet = histories[0]
        user_hs = histories[1: h_len+1]
        ment_hs = histories[h_len+1: 101]
        tweet, (_, _) = self.text_encoder(tweet)
        # tweet (b, 33, d_embed)
        tweet = tweet.mean(dim=1)
        # (b, d_embed)

        user_hs = [
            self.text_encoder(user_hs[i])[0].mean(dim=1)
            for i in range(len(user_hs))
        ]
        ment_hs = [
            self.text_encoder(ment_hs[i])[0].mean(dim=1)
            for i in range(len(ment_hs))
        ]
        # list of (b, d_embed)
        user_hs = torch.stack(user_hs, dim=1)
        ment_hs = torch.stack(ment_hs, dim=1)
        # (b, 50, d_embed)
        # user_hs = user_hs.sum(dim=1)
        # ment_hs = ment_hs.sum(dim=1)
        # user_len = u_mask.sum(dim=1) + 1e-25
        # ment_len = m_mask.sum(dim=1) + 1e-25
        # user_hs = user_hs / user_len
        # ment_hs = ment_hs / ment_len
        
        # (b, d_embed*3)
        return tweet, user_hs, ment_hs


class LSTMClassifier(nn.Module):
    def __init__(self, n_vocab, d_embed):
        super(LSTMClassifier, self).__init__()
        self.encoder = LSTMEncoder(n_vocab, d_embed)
        self.classifier = MLPClassifier(d_embed*3, 1)

    def forward(self, tweet, user_hs, ment_hs, u_mask=None, m_mask=None):
        tweet, user_hs, ment_hs = self.encoder(tweet, user_hs, ment_hs)
        user_hs = user_hs.mean(dim=1)
        ment_hs = ment_hs.mean(dim=1)
        feature = torch.cat([tweet, user_hs, ment_hs], dim=-1)
        logits = self.classifier(feature)

        return logits
