import torch
torch.use_deterministic_algorithms(False)
import sys
from models.TransformerLM import *
from data.BuildVocab import *
from spacy.tokenizer import Tokenizer
#torch.serialization.add_safe_globals([Vocabulary, Tokenizer])
from torch.distributions import Categorical
torch.backends.cudnn.deterministic = True

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

MAX_LENGTH = 500

def main():
   CHKPT_PATH = "./chkpts/Pnf4n0_GPT"
   chkpt = torch.load(CHKPT_PATH, weights_only=False)
   config = chkpt['config']

   print(CHKPT_PATH +" // "+str(chkpt['epoch']))
   # load vocab
   vocab = chkpt['vocab']

   # load model
   model = TransformerLM(len(vocab), config["d_model"], config["n_heads"], config["n_layers"])
   model.load_state_dict(chkpt['model_state_dict'])
   model.to(device)

   while True:
    # ask for prompt
    prompt = input("\n\nPrompt:\n")

    # numeralize prompt
    num_prompt = vocab.text2idx(prompt)
    l = len(num_prompt)


    for sampler in [argmaxDecode, sampleDecode, nucleusDecode]:
      torch.manual_seed(0)
      random.seed(0)
      torch.cuda.manual_seed(0)
      torch.cuda.manual_seed_all(0)

      src = torch.zeros(1,MAX_LENGTH)
      src[0,0] = 1 # <SOS>
      src[0,1:l+1] = torch.Tensor(num_prompt)
      src = src.to(dtype=int, device=device)
      print("\n\n")
      print(sampler)
      print(prompt, end="",flush=True)
      for t in range(l+1,MAX_LENGTH):
          out = model(src)

          src[0,t] =  sampler(out[:,t-1,:])

          w = vocab.idx2text([src[0,t].cpu().item()])[0]

          if w == "<EOS>":
              break
          if not any(x in w for x in [".",",","\"","'","!","?"]):
              w = " "+w

          print(w,  end='',flush=True)
      print("\n")
   sys.exit(1)


def argmaxDecode(scores):
    # TODO
    w_max = torch.argmax(scores, dim=-1).item()
    return w_max

def sampleDecode(scores, temp = 0.5):
   # TODO
   tmp_scores = scores / temp

   #probability using softmax
   prob = torch.softmax(tmp_scores, dim=-1)

   # random sample -> pick one
   w_sample = torch.multinomial(prob, num_samples=1).item()

   return w_sample

def nucleusDecode(scores, p=0.9, temp = 0.5):
    # TODO
    tmp_scores = scores / temp

    # probability using softmax
    prob = torch.softmax(tmp_scores, dim=-1).squeeze()  # Shape: (V,)

    # Sort in descending order,
    # to find smallest set of words with cumulative prob of p
    sorted_probs, sorted_indices = torch.sort(prob, descending=True)

    # cumulate the probs
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # find min idx that cumsum exceeds p
    cutoff_index = ((cumulative_probs >= p).nonzero(as_tuple=True)[0].min().item())

    #init
    top_p_probs = sorted_probs[:cutoff_index + 1]
    top_p_indices = sorted_indices[:cutoff_index + 1]

    # others -> 0
    top_p_distribution = torch.zeros_like(prob)
    top_p_distribution[top_p_indices] = top_p_probs

    # normalize
    top_p_distribution /= top_p_distribution.sum()

    # random sample -> pick one
    w_sample = torch.multinomial(top_p_distribution, num_samples=1).item()

    return w_sample


main()
