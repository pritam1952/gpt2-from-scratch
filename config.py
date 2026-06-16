GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,   
    "emb_dim": 64,          
    "n_heads": 4,
    "n_layers": 2,          
    "drop_rate": 0.0,       
    "qkv_bias": False
}

BATCH_SIZE = 32    
LEARNING_RATE = 3e-4
EPOCHS = 1         
DEVICE = "cpu"