import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import warnings
from tqdm import tqdm
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def setup_ddp(rank, world_size):
    """
    Setup for Distributed Data Parallel training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """
    Clean up DDP
    """
    dist.destroy_process_group()

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    total_loss = 0
    num_batches = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()
            num_batches += 1

            if count <= num_examples:
                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)
                
                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count >= num_examples:
                print_msg('-'*console_width)
                break
        
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    if writer:
        writer.add_scalar('validation loss', avg_loss, global_step)
        writer.flush()
        
        try:
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()
        except Exception as e:
            print_msg(f"Warning: Could not compute metrics: {e}")

    return avg_loss

def get_all_sentences(ds, lang, max_samples):
    count = 0
    for item in ds:
        if count >= max_samples:
            break
        
        text = item.get(lang)
        if text:
            yield text
            count += 1

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang} at {tokenizer_path}...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang, len(ds)), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer for {lang} built and saved.")
    else:
        print(f"Loading tokenizer for {lang} from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config, rank=0, world_size=1):
    max_samples = config.get('max_samples', 250000)
    
    if rank == 0:
        print(f"Loading {config['datasource']} dataset (English-Hindi)...")
    
    try:
        ds_full = load_dataset(
            config['datasource'],
            split='train'
        )
        
        if rank == 0:
            print(f"Dataset loaded. Total samples: {len(ds_full)}")
        
        ds_raw = []
        
        # FIXED: Handle None for max_samples properly
        if max_samples is None:
            total_to_load = len(ds_full)
            if rank == 0:
                print(f"Processing all {total_to_load} samples...")
        else:
            total_to_load = min(max_samples, len(ds_full))
            if rank == 0:
                print(f"Processing up to {total_to_load} samples...")
        
        with tqdm(total=total_to_load, desc="Loading samples", disable=(rank != 0)) as pbar:
            for idx, item in enumerate(ds_full):
                # Stop if we've reached max_samples (only if max_samples is not None)
                if max_samples is not None and len(ds_raw) >= max_samples:
                    break
                
                translation = item.get('translation', {})
                src_text = translation.get(config['lang_src'], '').strip()
                tgt_text = translation.get(config['lang_tgt'], '').strip()
                
                if src_text and tgt_text and len(src_text) > 3 and len(tgt_text) > 3:
                    ds_raw.append({
                        config['lang_src']: src_text,
                        config['lang_tgt']: tgt_text
                    })
                    pbar.update(1)
                
                # For None case, break when we've processed all samples
                if max_samples is None and idx >= len(ds_full) - 1:
                    break
        
        if rank == 0:
            print(f"Total valid samples collected: {len(ds_raw)}")
        
    except Exception as e:
        if rank == 0:
            print(f"Error loading dataset: {e}")
        raise RuntimeError(f"Could not load dataset: {e}")
    
    if len(ds_raw) == 0:
        raise RuntimeError(f"No valid data found for {config['lang_src']}-{config['lang_tgt']}.")
    
    if rank == 0:
        print("Building/loading tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    
    train_ds_raw = ds_raw[:train_ds_size]
    val_ds_raw = ds_raw[train_ds_size:]
    
    if rank == 0:
        print(f"Training samples: {len(train_ds_raw)}")
        print(f"Validation samples: {len(val_ds_raw)}")

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    if rank == 0:
        max_len_src = 0
        max_len_tgt = 0
        print("Checking max sentence lengths...")
        for item in tqdm(ds_raw[:1000], desc="Checking lengths"):
            src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
            tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f'Max length of source sentence (sampled): {max_len_src}')
        print(f'Max length of target sentence (sampled): {max_len_tgt}')
        print(f"Sequences will be truncated/padded to {config['seq_len']}")
    
    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, train_sampler

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        config["seq_len"], 
        config['seq_len'], 
        d_model=config['d_model'],
        N=config.get('N_layers', 6),
        h=config.get('n_heads', 8),
        dropout=config.get('dropout', 0.1),
        d_ff=config.get('d_ff', 2048)
    )
    return model

def train_on_gpu(rank, world_size, config):
    """
    Training function for each GPU process
    """
    # Setup DDP
    setup_ddp(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"🚀 DISTRIBUTED DATA PARALLEL (DDP) TRAINING")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {config['batch_size']}")
        print(f"Total batch size: {config['batch_size'] * world_size}")
        print(f"Gradient accumulation steps: {config.get('gradient_accumulation_steps', 1)}")
        print(f"Effective batch size: {config['batch_size'] * world_size * config.get('gradient_accumulation_steps', 1)}")
        print(f"{'='*80}\n")

    # Create model folder (only rank 0)
    if rank == 0:
        Path(f"{config['datasource'].replace('/', '_')}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, train_sampler = get_ds(config, rank, world_size)
    
    if rank == 0:
        print(f"Source vocab size: {tokenizer_src.get_vocab_size()}")
        print(f"Target vocab size: {tokenizer_tgt.get_vocab_size()}")

    # Build model and wrap with DDP
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model = DDP(model, device_ids=[rank])
    
    # Tensorboard (only rank 0)
    writer = SummaryWriter(config['experiment_name']) if rank == 0 else None

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # GradScaler for mixed precision
    scaler = GradScaler() if config.get('use_amp', True) else None

    initial_epoch = 0
    global_step = 0
    
    # Load checkpoint if exists (only rank 0 loads, then broadcast)
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    
    if model_filename and Path(model_filename).exists() and rank == 0:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        if scaler and 'scaler_state_dict' in state:
            scaler.load_state_dict(state['scaler_state_dict'])
        print(f"Resuming from epoch {initial_epoch}")
    
    # Synchronize all processes
    dist.barrier()

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('early_stopping_patience', 5)
    min_delta = config.get('early_stopping_min_delta', 0.001)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    if rank == 0:
        print(f"Early stopping: patience={patience}, min_delta={min_delta}")
        print(f"Mixed precision training: {config.get('use_amp', True)}")
        print(f"\nStarting training for {config['num_epochs']} epochs...")
        print("="*80)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        
        # Set epoch for DistributedSampler
        train_sampler.set_epoch(epoch)
        
        batch_iterator = tqdm(train_dataloader, desc=f"GPU{rank} Epoch {epoch:02d}/{config['num_epochs']}", disable=(rank != 0))
        
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            if config.get('use_amp', True):
                with autocast():
                    encoder_output = model.module.encode(encoder_input, encoder_mask)
                    decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.module.project(decoder_output)
                    
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
            else:
                encoder_output = model.module.encode(encoder_input, encoder_mask)
                decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.module.project(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            if rank == 0:
                batch_iterator.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:6.3f}"})
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if config.get('use_amp', True):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                
                if rank == 0 and writer:
                    writer.add_scalar('train loss', loss.item() * gradient_accumulation_steps, global_step)
                    writer.flush()
                
                global_step += 1

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        if rank == 0:
            print(f"\nEpoch {epoch:02d} - Average Training Loss: {avg_epoch_loss:.4f}")

        torch.cuda.empty_cache()
        dist.barrier()

        # Validation (only rank 0)
        if rank == 0:
            val_loss = run_validation(model.module, val_dataloader, tokenizer_src, tokenizer_tgt, 
                                       config['seq_len'], device, lambda msg: print(msg), 
                                       global_step, writer)

            print(f'Epoch {epoch:02d} - Validation Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_filename = get_weights_file_path(config, "best")
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'val_loss': val_loss
                }
                if scaler:
                    save_dict['scaler_state_dict'] = scaler.state_dict()
                torch.save(save_dict, best_model_filename)
                print(f'✓ Best model saved (val_loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                print(f'⚠ No improvement. Patience: {patience_counter}/{patience}')
                
                if patience_counter >= patience:
                    print(f'\n{"="*80}')
                    print(f'Early stopping at epoch {epoch + 1}')
                    print(f'Best validation loss: {best_val_loss:.4f}')
                    print(f'{"="*80}\n')
                    break

            # Save checkpoint
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }
            if scaler:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, model_filename)
            
            print(f"Checkpoint saved: {model_filename}\n")
        
        dist.barrier()
    
    if rank == 0:
        print(f'\nTraining completed!')
        print(f'Best validation loss: {best_val_loss:.4f}')
        print(f'Best model: {get_weights_file_path(config, "best")}')
    
    cleanup_ddp()

def train_model(config):
    """
    Main entry point - spawns processes for DDP
    """
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("WARNING: Less than 2 GPUs detected. Falling back to single GPU training.")
        # Fall back to single GPU training
        train_on_gpu(0, 1, config)
    else:
        print(f"Detected {world_size} GPUs. Starting DDP training...")
        # Spawn processes for each GPU
        mp.spawn(train_on_gpu, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
