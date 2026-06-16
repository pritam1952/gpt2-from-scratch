import time
import torch
import torch.nn.functional as F
import tiktoken

from torch.utils.data import DataLoader

from config import (
    GPT_CONFIG,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    DEVICE
)

from tokenizer import encode
from dataset import GPTDataset
from models import GPTModel


# ==================================================
# Loss Function
# ==================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = F.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


# ==================================================
# Evaluation
# ==================================================

@torch.no_grad()
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()

    train_losses = []
    val_losses   = []

    for i, (x, y) in enumerate(train_loader):
        if i >= eval_iter:
            break
        loss = calc_loss_batch(x, y, model, device)
        train_losses.append(loss.item())

    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iter:
            break
        loss = calc_loss_batch(x, y, model, device)
        val_losses.append(loss.item())

    model.train()

    return (
        sum(train_losses) / len(train_losses),
        sum(val_losses)   / len(val_losses)
    )


# ==================================================
# Text Generation
# ==================================================

@torch.no_grad()
def generate_text_simple(model, idx, max_new_tokens, context_size):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits   = model(idx_cond)
        logits   = logits[:, -1, :]

        probs    = torch.softmax(logits, dim=-1)

        # multinomial sampling (not greedy argmax)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


@torch.no_grad()
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()

    context_size = model.pos_emb.num_embeddings
    encoded      = text_to_token_ids(start_context, tokenizer).to(device)

    token_ids = generate_text_simple(
        model          = model,
        idx            = encoded,
        max_new_tokens = 50,
        context_size   = context_size
    )

    print("\nGenerated Sample:")
    print(token_ids_to_text(token_ids, tokenizer))
    print("-" * 80)

    model.train()


# ==================================================
# Training Loop
# ==================================================

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,           # ← added
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer
):
    train_losses     = []
    val_losses       = []
    tokens_seen_list = []

    tokens_seen = 0
    global_step = -1
    t0          = time.time()

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            loss.backward()

            # gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()     # ← cosine LR decay

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen_list.append(tokens_seen)

                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch+1} | "
                    f"Step {global_step:6d} | "
                    f"Train Loss {train_loss:.4f} | "
                    f"Val Loss {val_loss:.4f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e} | "
                    f"Time {elapsed/60:.1f}m"
                )

        # generate a sample at end of each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, tokens_seen_list


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("Loading dataset...")

    # train.py line ~140
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:   # ← fixed path
        text = f.read()

    tokens = encode(text)
    print(f"Total tokens: {len(tokens):,}")
    
    tokens = tokens[:200_000]
    
    split_idx    = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens   = tokens[split_idx:]

    train_dataset = GPTDataset(train_tokens, GPT_CONFIG["context_length"])
    val_dataset   = GPTDataset(val_tokens,   GPT_CONFIG["context_length"])

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Device    : {device}")
    print(f"Dataset   : {len(train_tokens):,} train  |  {len(val_tokens):,} val tokens")

    print("\nBuilding model...")
    model = GPTModel(GPT_CONFIG).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1,          # stronger regularisation
        betas=(0.9, 0.95)
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    print(f"Steps     : {total_steps:,}  ({len(train_loader):,}/epoch × {EPOCHS} epochs)")
    print("\nTraining started...\n")

    train_model_simple(
        model         = model,
        train_loader  = train_loader,
        val_loader    = val_loader,
        optimizer     = optimizer,
        scheduler     = scheduler,   # ← passed in
        device        = device,
        num_epochs    = EPOCHS,
        eval_freq     = 100,         # ← was 100, now 500
        eval_iter     = 5,          # ← was 10, now 20
        start_context = "To be or not to be",
        tokenizer     = tokenizer
    )

    torch.save(model.state_dict(), "gpt.pt")
    print("\nModel saved → gpt.pt")