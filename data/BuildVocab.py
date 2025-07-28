import os.path
import pickle

import re
from cleantext import clean
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
import spacy
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random
random.seed(0)

MAX_LEN = 128
MAX_EXAMPLES = 100

class Vocabulary:
  def __init__(self, corpus, tokenizer):
    self.tokenizer = tokenizer
    self.word2idx, self.idx2word = self.build_vocab(corpus)

  def __len__(self):
    return len(self.word2idx)

  def text2idx(self, text):
    tokens = [str(x).strip().lower() for x in self.tokenizer(text)]
    return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['<UNK>'] for t in tokens]

  def idx2text(self, idxs):
    return [self.idx2word[i] if i in self.idx2word.keys() else '<UNK>' for i in idxs]


  def build_vocab(self,corpus):
    cntr = Counter()
    for datapoint in tqdm(corpus):
      cntr.update( [str(x).strip().lower() for x in self.tokenizer(datapoint)] )

    tokens = [t for t,c in cntr.items() if c >= 1]
    word2idx = {t:i+4 for i,t in enumerate(tokens)}
    idx2word = {i+4:t for i,t in enumerate(tokens)}

    word2idx['<PAD>'] = 0  #add padding token
    idx2word[0] = '<PAD>'

    word2idx['<SOS>'] = 1  #add padding token
    idx2word[1] = '<SOS>'

    word2idx['<EOS>'] = 2  #add padding token
    idx2word[2] = '<EOS>'

    word2idx['<UNK>'] = 3  #add padding token
    idx2word[3] = '<UNK>'


    return word2idx, idx2word

def cleantext(text):
    text= clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=False,
        no_punct=False,
    )
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    return " ".join(text.split()) #Normlaize whitespace

# Cleaning function (lightweight)
def clean_text(example):
    text = example["text"].strip().lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = " ".join(text.split())        # Normalize whitespace

    example["text"] = text
    return example


class OpenWebTextDataset(Dataset):
  def __init__(self, split="train", vocab=None, force_reprocess=False):
    cache_path = f"./cached/openwebtext_cleaned_{split}"

    if os.path.exists(cache_path) and not force_reprocess:
        print(f"Loading cleaned dataset from {cache_path}")
        dataset = load_from_disk(cache_path)
    else:
        print("Loading raw data...")
        dataset = load_dataset("Skylion007/openwebtext", split=split, trust_remote_code=True)

        print("Cleaning dataset...")
        dataset = dataset.map(clean_text, num_proc=4, desc="Preprocessing Data")
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

        if len(dataset) > MAX_EXAMPLES:
            dataset = dataset.select(random.sample(range(len(dataset)), MAX_EXAMPLES))

        dataset.save_to_disk(cache_path)

    self.data = [x["text"] for x in dataset] 

    #self.data = [x["text"] for x in random.sample(list(dataset), MAX_EXAMPLES)]


    if vocab == None:
      print("Building vocab...")
      self.vocab = Vocabulary(self.data, spacy.load('en_core_web_sm').tokenizer)
    else:
      print("Loading Existing Vocab")
      self.vocab = vocab

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = self.vocab.text2idx(self.data[idx])
    l = min(MAX_LEN, len(x))
    numeralized = [self.vocab.word2idx['<SOS>']]+x[:l]+[self.vocab.word2idx['<EOS>']]
    return torch.tensor(numeralized)

  @staticmethod
  def pad_collate(batch):
      max_len = MAX_LEN + 2
      padded = []
      for x in batch:
          if len(x) < max_len:
              x = F.pad(x, (0, max_len - len(x)), value=0)
          else:
              x = x[:max_len]
          padded.append(x)
      return torch.stack(padded)

def save_vocab(vocab, file_path="./chkpts/vocab.pkl"):
  os.makedirs(os.path.dirname(file_path), exist_ok = True )
  with open(file_path, 'wb') as f:
    pickle.dump(vocab, f)
  print(f"Vocabulary saved to {file_path}")


def load_vocab(file_path="vocab.pkl"):
  with open(file_path, 'rb') as f:
    vocab = pickle.load(f)
  print(f"Vocabulary loaded from {file_path}")
  return vocab

def getOpenwebtextDataloadersAndVocab(batch_size=128, vocab_file = './chkpts/vocab.pkl'):

  if os.path.exists(vocab_file):
    print("Loading existing vocab!")
    vocab = load_vocab(vocab_file)
    train = OpenWebTextDataset(split="train", vocab=vocab)
  else:
    print("No existing vocabulary found. Building a new one...")
    train = OpenWebTextDataset(split="train")
    save_vocab(train.vocab, vocab_file)

  collate = OpenWebTextDataset.pad_collate
  train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=collate, drop_last=True)

  return train_loader, train.vocab


