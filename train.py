import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import signal
from functools import partial
import sys
import os
import numpy as np

from models.TransformerLM import *
from data.BuildVocab import *

from dotenv import load_dotenv

import datetime
import random
import string
import wandb
from tqdm import tqdm
from spacy.tokenizer import Tokenizer

from data.LoadTokenized import get_tokenized_dataloader
from data.bpeTonkenizer import *
#torch.serialization.add_safe_globals([Vocabulary, Tokenizer])
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

config = {
    "bs":64,   # batch size
    "lr":0.0003, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":10,
    "d_model":256,
    "n_heads":8,
    "n_layers":8
}



# This just runs a dummy batch through the model to see if anything breaks.
# Avoids having to load all the data before finding out the model is broken.
def dryRun():
  B = config["bs"]
  L = 300
  V = 8000
  input =  torch.randint(0, V, (B, L)).to(device)
  tmp_lm = TransformerLM(V, config["d_model"], config["n_heads"], config["n_layers"])
  tmp_lm.to(device)

  # Generate and inspect causal mask
  causal_mask = tmp_lm.generateCausalMask(L, input.device)  # [L, L]
  print("Causal mask (shape:", causal_mask.shape, ")")
  print(causal_mask.int())  # Print matrix

  plt.imshow(causal_mask.cpu().numpy(), cmap='Greys')
  plt.title("Causal Mask")
  plt.xlabel("Key Positions")
  plt.ylabel("Query Positions")
  plt.show()
  
  out = tmp_lm(input)

  loss = out.sum()
  loss.backward()
  
  assert out.shape == (B, L, V), "[Failed] Dry fit to check shapes work for a dummy input."
  print("[Passed] Dry fit to check shapes work for a dummy input.")
  
def main():
    # Quick sanity check before we do anything heavy
    dryRun()

    #load bin meta file
    
    dataset_name= 'openwebtext'
    max_examples = 800000
    block_size = 256
    bin_path, meta_path = prepare_tokenized_dataset(
       dataset_name=dataset_name, 
       max_examples=max_examples
    )

    train_loader, vocab_size = get_tokenized_dataloader(config["bs"], bin_path, meta_path, block_size)
    print("Vocab size (from tokenized data):", vocab_size)

    tmp_lm = TransformerLM(vocab_size, config["d_model"], config["n_heads"], config["n_layers"])
    print(tmp_lm)

    torch.compile(tmp_lm)

    train(tmp_lm, train_loader)


def train(model, train_loader):
    config["arch"] = str(model)
    run_name = generateRunName()

    # Startup wandb logging
    load_dotenv(dotenv_path="wandbkey.env")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(wandb_api_key)
    wandb.init(project="GPT-From-Scratch", name=run_name, config=config)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epoch"])

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    iteration = epoch = 0
    pbar = tqdm(total=config["max_epoch"] * len(train_loader), desc="Training Iterations", unit="batch")

    for epoch in range(config["max_epoch"]):
        signal.signal(signal.SIGINT, partial(interruptHandler,
                                             epoch, model, optimizer,
                                             scheduler, config, run_name))
        signal.signal(signal.SIGTERM, partial(interruptHandler,
                                              epoch, model, optimizer,
                                              scheduler, config, run_name))
        model.train()

        wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

        for x in train_loader:
            x = x.to(device)
            out = model(x)[:, :-1, :]
            x = x[:, 1:]

            loss = criterion(out.permute(0, 2, 1), x)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"Loss/train": loss.item()}, step=iteration)
            pbar.update(1)
            iteration += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config
        }, f"./chkpts/{run_name}")

        scheduler.step()

    wandb.finish()
    pbar.close()

       
def interruptHandler(epoch, model, optimizer, scheduler, config, run_name, sig, frame):
  torch.save({'epoch':epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict':optimizer.state_dict(),
              'scheduler_state_dict':scheduler.state_dict(),
              'config':config,
              },
              "./chkpts/"+run_name+"_interrupt")
  sys.exit(0)

def generateRunName():
  random.seed()
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_GPT"
  return run_name


main()