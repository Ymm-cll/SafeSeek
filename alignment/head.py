import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from torch.utils.data import Dataset
from typing import Dict, List, Any

class MaskedAttentionWrapper(nn.Module):
    use_inverse_mask = False
    
    def __init__(self, original_attn, num_attention_heads, num_key_value_heads, init_value=1.0):
        super().__init__()
        self.original_attn = original_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        for param in self.original_attn.parameters():
            param.requires_grad = False
        
        device = next(self.original_attn.parameters()).device
        dtype = next(self.original_attn.parameters()).dtype
        
        self.logits = nn.Parameter(torch.ones(num_attention_heads, device=device, dtype=dtype) * init_value)
        
        self.num_queries_per_kv = num_attention_heads // num_key_value_heads
    
    def get_mask(self, temperature=1.0, hard=True):
        logits_to_use = self.logits
        mask_soft = torch.sigmoid(logits_to_use / temperature)
        if hard:
            mask_hard = (mask_soft > 0.5).to(mask_soft.dtype)
            mask = mask_hard - mask_soft.detach() + mask_soft
        else:
            mask = mask_soft
        return mask
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                output_attentions=False, use_cache=False, temperature=1.0, **kwargs):
        head_mask = self.get_mask(temperature)
        if self.use_inverse_mask:
            head_mask = 1.0 - head_mask
        
        outputs = self.original_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        attn_output = outputs[0]
        head_dim = attn_output.shape[-1] // self.num_attention_heads
        expanded_mask = head_mask.repeat_interleave(head_dim)
        
        for _ in range(attn_output.dim() - 1):
            expanded_mask = expanded_mask.unsqueeze(0)
        
        masked_output = attn_output * expanded_mask
        return (masked_output,) + outputs[1:]
    
    @staticmethod
    def save_masks(model, path):
        mask_state = {}
        for name, module in model.named_modules():
            if isinstance(module, MaskedAttentionWrapper):
                mask_state[name] = module.logits.data.cpu()
        torch.save(mask_state, path)
        print(f"[MaskedAttentionWrapper] Masks saved to {path}")
    
    @staticmethod
    def load_masks(model, path, map_location="cpu"):
        loaded_masks = torch.load(path, map_location=map_location)
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, MaskedAttentionWrapper) and name in loaded_masks:
                module.logits.data.copy_(loaded_masks[name].to(module.logits.device))
                count += 1
        print(f"[MaskedAttentionWrapper] Loaded {count} masks from {path}")


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def patch_model(model):
    mask_params = []
    
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', num_attention_heads)
    
    print("Model configuration:")
    print(f"  - num_attention_heads: {num_attention_heads}")
    print(f"  - num_key_value_heads: {num_key_value_heads}")
    if num_key_value_heads != num_attention_heads:
        print(f"  - GQA detected: {num_attention_heads // num_key_value_heads} queries per KV head")
    
    for name, module in model.named_modules():
        if name.endswith('self_attn'):
            parent_name = ".".join(name.split(".")[:-1])
            parent = model.get_submodule(parent_name)
            
            wrapper = MaskedAttentionWrapper(module, num_attention_heads, num_key_value_heads)
            setattr(parent, 'self_attn', wrapper)
            
            mask_params.append(wrapper.logits)
    
    print(f"Patched {len(mask_params)} attention layers")
    return mask_params

class SafetyCircuitDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=256):
        tokenizer.padding_side = "right"
        self.raw_data = data_list 
        self.samples = []
        for item in data_list:
            scenarios = {
                'accept': (item['prompt'], item['accept']),
                'refuse': (item['prompt'], item['refuse'])
            }
            
            encoded_scenarios = {}
            for key, (prompt, completion) in scenarios.items():
                # Construct Chat format
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
                
                # Create prompt-only version to find the split point
                prompt_messages = [{"role": "user", "content": prompt}]
                prompt_full = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(prompt_full, add_special_tokens=False)
                
                # Create full sequence
                full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                full_enc = tokenizer(full_text, truncation=True, max_length=max_length, padding='max_length', return_tensors="pt")
                
                input_ids = full_enc["input_ids"].squeeze(0)
                attention_mask = full_enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                
                # Mask prompt and padding
                labels[:len(prompt_ids)] = -100
                labels[attention_mask == 0] = -100
                
                if key == 'accept':
                    eos_positions = (input_ids == tokenizer.eos_token_id)
                    labels[eos_positions] = -100
                
                encoded_scenarios[key] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
            self.samples.append(encoded_scenarios)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class CircuitDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        # Stack the nested dictionaries
        batch = {}
        for key in ['accept', 'refuse']:
            batch[key] = {
                'input_ids': torch.stack([f[key]['input_ids'] for f in features]),
                'attention_mask': torch.stack([f[key]['attention_mask'] for f in features]),
                'labels': torch.stack([f[key]['labels'] for f in features])
            }
        return batch

# --- Custom Trainer ---
class CircuitTrainer(Trainer):
    def __init__(
        self,
        mask_params,
        accept_weight=1.0,
        refuse_weight=1.0,
        sparsity_weight=1.0,
        output_dir=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_params = mask_params
        self.accept_weight = accept_weight
        self.refuse_weight = refuse_weight
        self.sparsity_weight = sparsity_weight
        self.output_dir_for_best = output_dir
        self.best_val_loss = None
        self.best_epoch = -1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Determine device
        device = next(model.parameters()).device
        
        # Helper to move nested dict to device
        def to_device(b):
            return {k: v.to(device) for k, v in b.items()}

        batch_accept = to_device(inputs['accept'])
        batch_refuse = to_device(inputs['refuse'])

        # Task 1: Accept Loss (Model, mask=1)
        MaskedAttentionWrapper.use_inverse_mask = False
        outputs_accept = model(**batch_accept)
        loss_accept = outputs_accept.loss

        # Task 2: Refuse Loss (Ablated Model, mask=0)
        MaskedAttentionWrapper.use_inverse_mask = True
        outputs_refuse = model(**batch_refuse)
        loss_refuse = outputs_refuse.loss

        # Task 3: Sparsity Loss
        loss_sparse = sum([(1.0 - torch.sigmoid(p)).mean() for p in self.mask_params])

        total_loss = (
            self.accept_weight * loss_accept +
            self.refuse_weight * loss_refuse +
            self.sparsity_weight * loss_sparse
        )

        # Cache last losses for epoch-level logging
        def _scalar(x):
            return x.detach().float().mean().item()
        self._last_loss_stats = {
            "total_loss": _scalar(total_loss),
            "loss_accept": _scalar(loss_accept),
            "loss_refuse": _scalar(loss_refuse),
            "sparsity_loss": _scalar(loss_sparse),
        }

        # Reset global state to default
        MaskedAttentionWrapper.use_inverse_mask = False

        # Note: We return outputs_accept as the canonical output, but the loss is composite
        return (total_loss, outputs_accept) if return_outputs else total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        # Custom eval step for nested inputs
        model.eval()
        device = next(model.parameters()).device

        def to_device(b):
            return {k: v.to(device) for k, v in b.items()}

        batch_accept = to_device(inputs["accept"])
        batch_refuse = to_device(inputs["refuse"])

        with torch.no_grad():
            MaskedAttentionWrapper.use_inverse_mask = False
            loss_accept = model(**batch_accept).loss

            MaskedAttentionWrapper.use_inverse_mask = True
            loss_refuse = model(**batch_refuse).loss

            loss_sparse = sum([(1.0 - torch.sigmoid(p)).mean() for p in self.mask_params])

            loss = (
                self.accept_weight * loss_accept +
                self.refuse_weight * loss_refuse +
                self.sparsity_weight * loss_sparse
            )

            # Cache last losses for eval logging
            def _scalar(x):
                return x.detach().float().mean().item()
            self._last_loss_stats = {
                "total_loss": _scalar(loss),
                "loss_accept": _scalar(loss_accept),
                "loss_refuse": _scalar(loss_refuse),
                "sparsity_loss": _scalar(loss_sparse),
            }

        MaskedAttentionWrapper.use_inverse_mask = False
        return (loss.detach(), None, None)

    def log(self, logs):
        # Suppress default Trainer log dicts; we print custom epoch logs instead.
        return

    def _compute_mask_stats(self, model):
        stats = {"total": 0, "zeros": 0}
        threshold = 0.5
        for _, module in model.named_modules():
            if isinstance(module, MaskedAttentionWrapper):
                mask = (torch.sigmoid(module.logits) > threshold).float()
                stats["total"] += mask.numel()
                stats["zeros"] += (mask == 0).sum().item()
        return stats

    def _format_mask_stats(self, stats):
        total = stats["total"]
        zeros = stats["zeros"]
        sparsity = (zeros / total * 100.0) if total > 0 else 0.0
        active = total - zeros
        return [
            "[Attention Head Mask Sparsity]",
            f"  Active heads    : {active:4d}/{total:4d} ({100-sparsity:5.2f}%)",
            f"  Inactive heads  : {zeros:4d}/{total:4d} ({sparsity:5.2f}%)",
        ]

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Print detailed loss breakdown
        if hasattr(self, '_last_loss_stats') and self._last_loss_stats:
            print("\n[Eval Loss Breakdown]", flush=True)
            print(f"  Total loss     : {self._last_loss_stats['total_loss']:.6f}", flush=True)
            print(f"  Accept loss    : {self._last_loss_stats['loss_accept']:.6f}", flush=True)
            print(f"  Refuse loss    : {self._last_loss_stats['loss_refuse']:.6f}", flush=True)
            print(f"  Sparsity loss  : {self._last_loss_stats['sparsity_loss']:.6f}", flush=True)

            # Check for NaN values
            nan_components = []
            for key, val in self._last_loss_stats.items():
                if np.isnan(val):
                    nan_components.append(key)
            if nan_components:
                print(f"\n WARNING: NaN detected in: {', '.join(nan_components)}", flush=True)

        stats = self._compute_mask_stats(self.model)
        lines = self._format_mask_stats(stats)
        for line in lines:
            print(line, flush=True)

        # Check and save best epoch (based on accept + refuse loss only, no sparsity)
        if hasattr(self, '_last_loss_stats') and self._last_loss_stats and self.output_dir_for_best:
            # Compute validation loss (only accept + refuse, no sparsity)
            val_loss = (
                self.accept_weight * self._last_loss_stats['loss_accept'] +
                self.refuse_weight * self._last_loss_stats['loss_refuse']
            )

            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = int(self.state.epoch) if self.state and self.state.epoch is not None else -1

                # Save masks
                save_path = os.path.join(self.output_dir_for_best, "best_safety_masks.pt")
                MaskedAttentionWrapper.save_masks(self.model, save_path)
                print(f"\n✓ Best model saved at epoch {self.best_epoch} (val_loss={val_loss:.6f})", flush=True)

        print("", flush=True)  # Add blank line for readability
        return metrics

@torch.no_grad()
def generate_and_save_results(model, tokenizer, val_data, batch_size=8, output_file="generation_results.json", use_inverse_mask=False):
    tokenizer.padding_side = "left"
    model.eval()
    results = []
    
    MaskedAttentionWrapper.use_inverse_mask = use_inverse_mask
    
    for i in tqdm(range(0, len(val_data), batch_size), desc=f"Generating results (Inverse={use_inverse_mask})"):
        batch_items = val_data[i : i + batch_size]
        
        def get_templated_prompts(prompts):
            templated = []
            for p in prompts:
                msg = [{"role": "user", "content": p}]
                templated.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            return templated

        prompts_raw = [item['prompt'] for item in batch_items]
        prompts = get_templated_prompts(prompts_raw)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        responses = []
        for j in range(len(prompts)):
            input_len = inputs.input_ids[j].shape[0]
            resp = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            responses.append(resp)
        
        for idx, item in enumerate(batch_items):
            results.append({
                "prompt": item['prompt'],
                "refuse": item['refuse'],
                "accept": item['accept'],
                "generation": responses[idx]
            })
            
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Generation results saved to {output_file}")


# --- Training Logic ---
def train_circuits():
    MODEL_PATH = "YOUR MODEL PATH"
    OUTPUT_DIR = "YOUR OUTPUT DIRECTORY"
    BATCH_SIZE = 8
    
    set_random_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Load dataset
    with open("YOUR DATASET PATH", "r") as f:
        advbench_data = json.load(f)
    
    full_data = advbench_data['results']
    train_data = full_data[:100]
    val_data = full_data[100:150]
    
    train_dataset = SafetyCircuitDataset(train_data, tokenizer)
    val_dataset = SafetyCircuitDataset(val_data, tokenizer)
    
    # Patch model
    mask_params = patch_model(model)

    print(f"Number of mask parameters to optimize: {len(mask_params)}")
    
    # === Loss weights ===
    accept_weight = 2.0
    refuse_weight = 2.0
    sparsity_weight = 0.1
    
    print(
        "Loss weights:",
        f"accept={accept_weight}, refuse={refuse_weight}, sparsity={sparsity_weight}"
    )
    
    # --- Hugging Face Trainer Setup ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=50,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=1e-2,
        logging_strategy="no",
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        label_names=[],
    )

    trainer = CircuitTrainer(
        mask_params=mask_params,
        accept_weight=accept_weight,
        refuse_weight=refuse_weight,
        sparsity_weight=sparsity_weight,
        output_dir=OUTPUT_DIR,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CircuitDataCollator(),
    )

    trainer.train()

    # Print best epoch info
    print(f"\n{'='*80}")
    print(f"Training completed!")
    if trainer.best_epoch >= 0:
        print(f"Best epoch: {trainer.best_epoch} (val_loss={trainer.best_val_loss:.6f})")
        print(f"Best model saved to: {OUTPUT_DIR}/best_safety_masks.pt")
    else:
        print(f"No best model saved")
    print(f"{'='*80}\n")

    # Save final masks
    MaskedAttentionWrapper.save_masks(model, f"{OUTPUT_DIR}/final_safety_masks.pt")
    
    # Post-training generation
    generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, f"{OUTPUT_DIR}/post_mask_accept.json", use_inverse_mask=False)
    generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, f"{OUTPUT_DIR}/post_mask_refuse.json", use_inverse_mask=True)

    print("Training completed.")

if __name__ == "__main__":
    train_circuits()