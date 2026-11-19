# Adventure Story Weaver

A small GPT-style story continuation model trained from scratch on the included original fantasy corpus. Implements byte-level BPE, embeddings, causal attention, transformer blocks, training and top-k sampling.

## Run on Windows

1. Install Python 3.10+ from python.org and check **Add Python to PATH**.
2. Extract this folder.
3. Open Command Prompt in this folder.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Train:

   ```bash
   python train.py --steps 1500
   ```

   Training time depends on your computer. The model is deliberately small; a GPU is helpful but not required.

6. Generate stories:

   ```bash
   python generate.py
   ```

   Or one prompt at a time:

   ```bash
   python generate.py --prompt "Mira entered the forgotten castle" --tokens 200
   ```

## Commands / tuning

- `python train.py --steps 2000 --lr 0.0003 --batch-size 32`
- `python generate.py --tokens 250 --temperature 0.8 --top-k 30`
- Type `quit` in the interactive generator to exit.

## Project files

- `corpus.txt`: sample original fantasy narrative used for training
- `tokenizer.py`: byte-level BPE tokenizer and merge learning
- `model.py`: token/position embeddings, causal self-attention and transformer blocks
- `train.py`: batches, next-token training, AdamW and loss reporting
- `generate.py`: interactive CLI, temperature and top-k sampling
- `tokenizer.json`, `artifacts/model.pt`: created by training

## Learning note

This is a tiny demonstration model, not a pretrained language model. Its output may be repetitive or inconsistent, especially with a short corpus. Improve results by adding more legally reusable adventure/fantasy text and training longer. The corpus included here is an original miniature sample, not a copied novel.
