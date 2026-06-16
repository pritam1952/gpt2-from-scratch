import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def encode(text):
    return tokenizer.encode(text)

def decode(tokens):
    return tokenizer.decode(tokens)