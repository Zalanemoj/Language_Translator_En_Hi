from pathlib import Path

def get_config():
    return {
        # MULTI-GPU OPTIMIZED SETTINGS
        "batch_size": 32,  # Per GPU batch size (total = 16 * 2 = 32)
        "gradient_accumulation_steps": 2,  # Effective batch size = 32 * 2 = 64
        "num_epochs": 100,
        "lr": 10**-4,
        "seq_len": 256,
        "d_model": 512,
        "datasource": 'cfilt/iitb-english-hindi',
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.001,
        "max_samples": 250000,
        "N_layers": 6,  # Increased back to 6 with 2 GPUs
        "n_heads": 8,
        "dropout": 0.1,
        "d_ff": 2048,
        "use_amp": True,
        "num_workers": 4,  # Increased for multi-GPU
        "pin_memory": True,
        # Multi-GPU settings
        "use_multi_gpu": True,
        "gpu_ids": [0, 1]  # Use both GPUs
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource'].replace('/', '_')}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
