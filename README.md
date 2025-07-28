Project Blog URL: https://woonki94.pages.dev/Projects/GPT2-From-Scratch/

Run train.py : it will build vocab from openwebtext, and train the transformer and embedding

To generate the text, run generate.py



with 250000 MAX_EXAMPLES in buildVocab, ran into GPU ERROR -> need to figure out to make memory safe model or training ,,


---

### Why `Vocabulary`-based code causes **GPU OOM**:

First it builds a **custom vocabulary** and tokenizes + batches data **at runtime** like this:

1. **Reads all raw text lines** into memory (`self.data = [x["text"] for x in dataset]`)
2. Each `__getitem__` dynamically:

   * Tokenizes the text
   * Converts it to indices (`text2idx`)
   * Adds `<SOS>`, `<EOS>`, and pads with `pad_collate`
3. **Each batch can have variable-length inputs**, padded up to a `MAX_LEN` (e.g., 128)
4. The full batch of size `B × L × d_model` is sent to the model on GPU

Now, here’s **why OOM happens**:

---

#### 1. Tokenizing on-the-fly eats CPU & RAM

tokenizing and building sequences **in real time** per batch. This adds CPU load and RAM usage, which indirectly slows down training and can overflow system memory before GPU is even used.

---

#### 2. vocabulary is large & grows with dataset

If usign a bigger dataset (e.g., OpenWebText), vocabulary grows significantly, e.g., from hundreds to **tens of thousands of tokens**. This increases:

* The size of the embedding matrix (`vocab_size × d_model`)
* The logits (`B × L × vocab_size`) output by the model
* Memory for gradients & attention computations
 
* For example, a `vocab_size = 50,000` and `batch_size = 64`, `seq_len = 128`, `d_model = 512` can easily exceed VRAM if multiple layers are used.

---

#### 3. Longer sequences or examples increase memory quadratically

The **attention mechanism** has complexity of `O(L²)`:

* A sequence of 128 tokens → `128×128` attention matrix per head
* With 8 heads and 64 examples = `64 × 8 × 128 × 128` → **over 8 million float entries**, just for attention weights

If using **larger sequences or more examples**, it can OOM easily.

---

#### Solution: Tokenize and preprocess beforehand

Instead of `Vocabulary` and `text2idx` at runtime, switch to:

1. **Pre-tokenizing dataset** (e.g., with `tiktoken` or `transformers`)
2. Saving tokenized examples to disk
3. Loading them in a fixed format (e.g., `input_ids` tensors)
4. Feeding them directly into the model

This allows:

* **Efficient batching**
* **Less CPU/RAM usage**
* **Faster training**
* **Less risk of OOM**

---

