import os
import re
import numpy as np
import pickle
from tqdm import tqdm
from datasets import load_dataset, Dataset
import tiktoken

# === Config ===
split = "train"
num_proc = 4
max_examples = 100  # Example: 100_000 for full data
force_reprocess = False  # Set to True to force reprocessing even if output exists



# === Paths ===
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.abspath(os.path.join(base_dir, "../cached/openwebtext_tokenized"))
os.makedirs(output_dir, exist_ok=True)

tokenized_bin_file = os.path.join(output_dir, f"cleaned_data_{max_examples}.bin")
meta_file = os.path.join(output_dir, f"meta_{max_examples}.pkl")
cleaned_data_path = os.path.join(output_dir, f"cleaned_dataset_{max_examples}")

# === Early exit if output already exists ===
if not force_reprocess and os.path.exists(tokenized_bin_file) and os.path.exists(meta_file):
    print(f"Tokenized data already exists at {tokenized_bin_file}. Skipping processing.")
    exit(0)

# === Cleaning Function ===
def clean_text(example):
    text = example["text"].strip().lower()
    text = re.sub(r"<[^>]+>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove non-alphanumeric
    text = " ".join(text.split())  # normalize whitespace
    example["text"] = text
    return example

# === Tokenizer ===
enc = tiktoken.get_encoding("gpt2")

def tokenize(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

# === Load or Clean Dataset ===
if os.path.exists(cleaned_data_path) and not force_reprocess:
    print(f"Loading cached cleaned dataset from {cleaned_data_path}...")
    dataset = Dataset.load_from_disk(cleaned_data_path)
else:
    print("Loading and cleaning raw dataset...")
    dataset = load_dataset("Skylion007/openwebtext", split=split, trust_remote_code=True)
    if max_examples:
        dataset = dataset.select(range(min(len(dataset), max_examples)))
    dataset = dataset.map(clean_text, num_proc=num_proc)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    dataset.save_to_disk(cleaned_data_path)
    print(f"Saved cleaned dataset to {cleaned_data_path}")

# === Tokenization ===
print("Tokenizing dataset...")
tokenized = dataset.map(
    tokenize,
    remove_columns=["text"],
    num_proc=num_proc,
    desc="Tokenizing",
)

# === Save binary file ===
arr_len = np.sum(tokenized["len"], dtype=np.uint64)
dtype = np.uint16
arr = np.memmap(tokenized_bin_file, dtype=dtype, mode="w+", shape=(arr_len,))

print(f"Writing binary tokens to {tokenized_bin_file}...")
idx = 0
for record in tqdm(tokenized, desc="Writing tokens"):
    arr[idx : idx + record["len"]] = np.array(record["ids"], dtype=dtype)
    idx += record["len"]
arr.flush()

# === Save Metadata ===
meta = {
    "vocab_size": enc.n_vocab,
    "encoder": enc.name,
    "eot_token": enc.eot_token,
    "max_examples": max_examples,
}
with open(meta_file, "wb") as f:
    pickle.dump(meta, f)

print(f"\nSaved tokenized binary and metadata to: {output_dir}")
