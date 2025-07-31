

Project Blog URL: https://woonki94.pages.dev/Projects/GPT2-From-Scratch/

Run train.py : it will build vocab from openwebtext, and train the transformer and embedding

To generate the text, run generate.py


---

## TODO

### Core Pipeline
- [x] Dataset cleaning and tokenization
- [x] Save tokenized data as `.bin` using `np.memmap`
- [x] Build custom GPT model architecture
- [x] Training loop (from scratch)
- [x] Checkpoint saving and resuming
- [x] Tokenizer metadata and vocab tracking

### Finetuning
- [ ] Load custom model from scratch-trained checkpoint
- [ ] Finetune on new dataset using same tokenizer
- [ ] Modular finetune config (LR, warmup, max_iters)
- [ ] Instruction-style dataset finetuning (prompt → response)
- [ ] Add evaluation during finetuning (loss/perplexity)

### LoRA + Parameter Efficient Tuning
- [ ] Integrate LoRA with HuggingFace `peft`
- [ ] Manually inject LoRA into custom model (for full control)
- [ ] Save and merge LoRA adapters separately
- [ ] Add support for multi-LoRA adapter stacking

### Prompt Tuning
- [ ] Add soft prompt tuning support (via `peft`)
- [ ] Allow fixed model + trainable prompt embeddings
- [ ] Benchmark vs LoRA on small tasks

### Retrieval-Augmented Generation (RAG)
- [ ] Integrate FAISS or Qdrant vector DB
- [ ] Create document embedding pipeline
- [ ] Implement query + context prompt composer
- [ ] Connect RAG to inference loop

### Memory & Performance
- [ ] Add FlashAttention / Triton kernels
- [ ] Implement token packing for efficient batching
- [ ] Add gradient checkpointing
- [ ] Support mixed-precision (FP16 / bfloat16)
- [ ] Add optimizer & scheduler configs (cosine, warmup, etc.)

### Evaluation
- [ ] Perplexity computation on held-out data
- [ ] LAMBADA / PIQA / zero-shot benchmarks
- [ ] Loss + accuracy logging (TensorBoard or wandb)
- [ ] Compare full fine-tune vs LoRA vs prompt tuning

### Inference & Deployment
- [ ] Streaming generation loop
- [ ] Top-k / top-p sampling + temperature control
- [ ] FastAPI / Gradio interface
- [ ] Quantized model inference (4-bit / 8-bit)
- [ ] Model export to GGUF / ONNX

###  Documentation
- [ ] Setup instructions
- [ ] Training / finetuning examples
- [ ] Inference examples
- [ ] LoRA + RAG integration guide



