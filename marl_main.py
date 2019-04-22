import os
import time
import argparse
import logging
import random

from utilities.data_helper import load_data
from utilities.utils import *
from models.marl import MultiAgentClassifier, SMultiAgentClassifier
from models.batch_marl import BatchMultiAgentClassifier
from models.mean_marl import MeanMultiAgentClassifier
from trainers.ma_trainer import train_marl, valid_epoch
from trainers.ma_trainer import benchmark_epoch
# from trainers.loop_trainer import train_marl, valid_epoch
# from trainers.loop_trainer import benchmark_epoch
from trainers.ma_tester import test


torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

# ----------Preparing----------
parser = argparse.ArgumentParser(prog="text rl mention")
parser.add_argument("--debug", action="store_true",
                    help="to work in debug mode")
parser.add_argument("--device", default="cpu",
                    help="specialize which gpu to use, like 'cuda:0'")
parser.add_argument("--l2", type=float, default=1e-8,
                    help="set a float value for l2 regularization")
parser.add_argument("--batch_size", type=int, default=1,
                    help="set batch size for training and fine-tine data")
parser.add_argument("--d_embed", type=int, default=200,
                    help="dimensions of word embeddings")
parser.add_argument("--e_drop", type=float, default=0.2,
                    help="embedding dropout p")
parser.add_argument("--model_name", default="test",
                    help="specify a name for store model")
parser.add_argument("--dan", action="store_true",
                    help="use DAN or not")
# parser.add_argument("--gru", action="store_true",
#                     help="use gru or not")
parser.add_argument("--exp_name", default="test")
args = parser.parse_args()

# if args.debug:
#     data_path = "Dataset/debug_data_200.pkl"
# else:
#     data_path = "Dataset/splited_data_200.pkl"

device = torch.device(args.device)
logging.basicConfig(
    filename=f"log/{args.model_name}", filemode='w',
    format="%(message)s",
    level=logging.DEBUG
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger("").addHandler(console)

# ----------Loading data---------
# train_data, valid_data, test_data, \
#     n_vocab, n_sent, len_sent = load_data(args.batch_size)


class SentFeatures():
    """A single set of feature of a sentence"""

    def __init__(self, input_ids, input_mask, segment_ids,
                 text=None, sent_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.text = text
        self.sent_id = sent_id

train_data, valid_data, test_data, select_data = load_data(
    args.batch_size, args.debug
)

feature_path = "Dataset/word_feature_tensor.pkl"
with open(feature_path, 'rb') as f:
    feature = torch.load(f)
embedding = torch.nn.Embedding(feature.size(0), feature.size(1))
embedding.weight = torch.nn.Parameter(feature, requires_grad=False)
r"""
train_data:
    iterable of training data
valid_data:
    iterable of validation data
test_data:
    iterable of (train_part, test_part) tuple in testing data
    train_part is a iterable like train_data
    test_part is a iterable like valid_data
n_vocab:
    size of vocabulary (contain the pad_idx)
n_sent:
    number of total sentences in user history
len_sent:
    the truncated sentence length of sentences in user history
"""

args.n_vocab = 50001
# HARDCODE 50
args.n_sent = 50
# args.len_sent = len_sent

# ----------Configuring models----------
print(f"L2 regularization coefficient is set to {args.l2}")
print(f"There are total {args.n_sent} sentences in each user history")
print(f"Vocabulary size is {args.n_vocab}")
print(f"Embedding dimension is {args.d_embed}")
print(f"Embedding dropout p is {args.e_drop}")
print(f"Using device {device}")

if args.batch_size == 1:
    model = MultiAgentClassifier(args.n_vocab, args.d_embed, device)
elif args.dan:
    model = MeanMultiAgentClassifier(
        args.n_vocab, args.d_embed,
        args.batch_size, args.n_sent, device, "dan"
    )
    print("Using batched DAN model")
# elif args.gru:
#     model = MeanMultiAgentClassifier(
#         args.n_vocab, args.d_embed,
#         args.batch_size, args.n_sent, device, "gru"
#     )
#     print("Using batched gru model")
else:
    model = BatchMultiAgentClassifier(
        args.n_vocab, args.d_embed,
        args.batch_size, args.n_sent, device
    )
    print("Using batched model")

model_name = args.model_name

# ----------Benchmark----------
print("Test result for classifier")
# benchmark_epoch(embedding, model, valid_data, device)
# benchmark_epoch(embedding, model, select_data, device)
benchmark_epoch(embedding, model, test_data, device)

# ----------Training----------
system_start = time.time()
logging.info(f"System training start at {time.ctime(system_start)}")

# best_f1 = train_marl(
#     embedding, model, train_data, valid_data, test_data,
#     args.l2, device, model_name
# )

best_f1 = train_marl(
    embedding, model, train_data, select_data, valid_data, test_data,
    args.l2, device, model_name
)

logging.info("Joint training stopped at best f1 = {:.4f}".format(best_f1))

# ----------Testing----------
logging.info("System test start:")
model = torch.load(f"parameter/{model_name}")
model.to(device)
valid_epoch(embedding, model, test_data, device)
# test(test_data, args, device, fine_tune=False)
# test(test_data, args, device, fine_tune=True)

total_time_cost = time.time() - system_start
logging.info(f"Total time cost {dhms_time(total_time_cost)}")
