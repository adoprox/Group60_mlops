import numpy as np
import pandas as pd
from pathlib import Path
from typing import *

from sklearn.model_selection import train_test_split
from fastai.test_utils import *

# Importing necessary libraries from fastai v2
from fastai.text.all import *

# Importing the transformers library
from transformers import BertTokenizer, BertForSequenceClassification

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=False,
    bert_model_name="bert-base-uncased",
    max_lr=3e-5,
    epochs=4,
    use_fp16=True,
    bs=32,
    discriminative=False,
    max_seq_len=256,
)

# Using the transformers library for the tokenizer
bert_tok = BertTokenizer.from_pretrained(config.bert_model_name)

def is1d(arr: np.ndarray):
    return np.array(arr).ndim == 1

def _join_texts(texts:Collection[str], mark_fields:bool=False, sos_token:Optional[str]=BOS):
    """Adapted from fast.ai source for v2"""
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    text_col = f'{FLD} {1} ' + df[0].astype(str) if mark_fields else df[0].astype(str)
    if sos_token is not None: text_col = f"{sos_token} " + text_col
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)
    return text_col.values

class FastAiBertTokenizer(Transform):
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def encodes(self, t:str) -> List[int]:
        return self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len]

    def decodes(self, tokens:List[int]) -> str:
        return self.tokenizer.decode(tokens)

DATA_ROOT = Path("./Group60_mlops/data/raw")

train, test = [pd.read_csv(DATA_ROOT / fname) for fname in ["train.csv", "test.csv"]]
val = train # You can create a validation set using train_test_split

if config.testing:
    train = train.head(1024)
    val = val.head(1024)
    test = test.head(1024)

fastai_bert_vocab = L(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer.from_df(text_cols="comment_text", res_col_name="text", tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len))

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Creating the DataLoaders
dls = DataBlock(
    blocks=(TextBlock.from_df("comment_text", seq_len=config.max_seq_len, vocab=fastai_bert_vocab, 
                              tok=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len)), MultiCategoryBlock),
    get_x=ColReader("text"),
    get_y=ColReader(label_cols),
    splitter=RandomSplitter()
).dataloaders(train, bs=config.bs)

bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=6)

loss_func = nn.BCEWithLogitsLoss()

# Creating a Learner
learner = Learner(dls, bert_model, loss_func=loss_func, metrics=[accuracy_multi])

if config.use_fp16:
    learner = learner.to_fp16()

#learner.lr_find()

#learner.recorder.plot()

# learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)
    
learner.fit(n_epoch=config.epochs, lr=config.max_lr)
