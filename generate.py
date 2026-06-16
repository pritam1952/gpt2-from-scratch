
import torch
import tiktoken

from config import GPT_CONFIG
from models import GPTModel


@torch.no_grad()
def generate_text(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.8
):
    model.eval()

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(
            logits / temperature,
            dim=-1
        )

        idx_next = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1
        )

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)

    return torch.tensor(
        encoded,
        dtype=torch.long
    ).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(
        token_ids.squeeze(0).tolist()
    )


def main():

    device = torch.device("cpu")

    tokenizer = tiktoken.get_encoding("gpt2")

    print("Loading model...")

    model = GPTModel(GPT_CONFIG)

    model.load_state_dict(
        torch.load(
            "gpt.pt",
            map_location=device
        )
    )

    model.to(device)

    prompt = input("Enter prompt: ")

    input_ids = text_to_token_ids(
        prompt,
        tokenizer
    ).to(device)

    output_ids = generate_text(
        model=model,
        idx=input_ids,
        max_new_tokens=100,
        context_size=GPT_CONFIG["context_length"],
        temperature=0.8
    )

    generated_text = token_ids_to_text(
        output_ids,
        tokenizer
    )

    print("\nGenerated Text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()

