import os
import re
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
import tiktoken

# === Tokenizer ===
enc = tiktoken.get_encoding("gpt2")

def tokenize(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

# === Cleaning Function ===
def clean_text(example):
    text = example["text"].strip().lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join(text.split())
    example["text"] = text
    return example

def prepare_tokenized_dataset(
    dataset_name='openwebtext',
    split='train',
    max_examples=800000,
    num_proc=8,
    force_reprocess=False,
    base_dir=None,
):
    # Setup paths
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(base_dir, "../cached"))
    os.makedirs(output_dir, exist_ok=True)

    tokenized_bin_file = os.path.join(output_dir, f"tokenized_data__{dataset_name}_{max_examples}.bin")
    meta_file = os.path.join(output_dir, f"meta_{dataset_name}_{max_examples}.pkl")
    cleaned_data_path = os.path.join(output_dir, f"cleaned_{dataset_name}_{max_examples}")

    if not force_reprocess and os.path.exists(tokenized_bin_file) and os.path.exists(meta_file):
        print(f"Tokenized data already exists at {tokenized_bin_file}. Skipping processing.")
        return tokenized_bin_file, meta_file

 

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

    # === Tokenize ===
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Tokenizing"
    )

    # === Write binary ===
    arr_len = np.sum(tokenized["len"], dtype=np.uint64)
    dtype = np.uint16
    arr = np.memmap(tokenized_bin_file, dtype=dtype, mode="w+", shape=(arr_len,))

    print(f"Writing binary tokens to {tokenized_bin_file}...")
    idx = 0
    for record in tqdm(tokenized, desc="Writing tokens"):
        arr[idx : idx + record["len"]] = np.array(record["ids"], dtype=dtype)
        idx += record["len"]
    arr.flush()

    # === Save metadata ===
    meta = {
        "vocab_size": enc.n_vocab,
        "encoder": enc.name,
        "eot_token": enc.eot_token,
        "max_examples": max_examples,
    }
    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved tokenized binary and metadata to: {output_dir}")
    return tokenized_bin_file, meta_file

