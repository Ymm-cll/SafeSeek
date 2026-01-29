import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from peft import PeftModel
from torch.utils.data import Dataset
from typing import Dict, List, Any

class AblatedLinear(nn.Module):
    use_inverse_mask = False  # False: main path (G); True: circuit path (H)
    
    def __init__(self, original_layer, init_value=1.0):
        super().__init__()
        self.original_layer = original_layer
        
        # Ensure the original layer parameters are frozen
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        out_features = original_layer.out_features
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
        self.logits_main = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype) * init_value)
        self.logits_circuit = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype) * init_value)

    def get_mask(self, logits, temperature=1.0, hard=True):
        logits_to_use = logits
        mask_soft = torch.sigmoid(logits_to_use / temperature)
        if hard:
            # Straight-Through Estimator (STE)
            mask_hard = (mask_soft > 0.5).to(mask_soft.dtype)
            mask = mask_hard - mask_soft.detach() + mask_soft
        else:
            mask = mask_soft
        return mask.to(self.original_layer.weight.dtype)

    def forward(self, x):
        output = self.original_layer(x)
        if not self.use_inverse_mask:
            mask = self.get_mask(self.logits_main)
        else:
            mask = self.get_mask(self.logits_circuit)
        return output * mask

    @staticmethod
    def save_masks(model, path):
        mask_state = {}
        for name, module in model.named_modules():
            if isinstance(module, AblatedLinear):
                mask_state[name + ".logits_main"] = module.logits_main.data.cpu()
                mask_state[name + ".logits_circuit"] = module.logits_circuit.data.cpu()
        torch.save(mask_state, path)
        print(f"[AblatedLinear] Masks saved to {path}")

    @staticmethod
    def load_masks(model, path, map_location="cpu"):
        loaded_masks = torch.load(path, map_location=map_location)
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, AblatedLinear):
                key_main = name + ".logits_main"
                key_circuit = name + ".logits_circuit"
                if key_main in loaded_masks:
                    module.logits_main.data.copy_(loaded_masks[key_main].to(module.logits_main.device))
                    count += 1
                if key_circuit in loaded_masks:
                    module.logits_circuit.data.copy_(loaded_masks[key_circuit].to(module.logits_circuit.device))
                    count += 1
        print(f"[AblatedLinear] Loaded {count // 2} layers' masks from {path}")


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Model Patching Tool ---
def patch_model(model):
    layers_to_patch = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    main_mask_params = []
    circuit_mask_params = []
    for name, module in model.named_modules():
        for target in layers_to_patch:
            if name.endswith(target):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                new_module = AblatedLinear(module)
                setattr(parent, attr_name, new_module)
                main_mask_params.append(new_module.logits_main)
                circuit_mask_params.append(new_module.logits_circuit)
    return main_mask_params, circuit_mask_params

class BackdoorCircuitDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        tokenizer.padding_side = "right"
        self.raw_data = data_list 
        self.samples = []
        for item in data_list:
            scenarios = {
                'main_clean': (item['input'], item['output']),
                'main_backdoor': (item['backdoor_input'], item['output']),
                'circuit_clean': (item['input'], item['backdoor_output']),
                'circuit_backdoor': (item['backdoor_input'], item['backdoor_output']),
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

# --- Custom Data Collator ---
class CircuitDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        # Stack the nested dictionaries
        batch = {}
        for key in ['main_clean', 'main_backdoor', 'circuit_clean', 'circuit_backdoor']:
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
        main_mask_params,
        circuit_mask_params,
        main_clean_weight=1.0,
        main_backdoor_weight=1.0,
        circuit_backdoor_weight=1.0,
        circuit_clean_weight=1.0,
        main_sparsity_weight=1.0,
        circuit_sparsity_weight=1.0,
        overlap_weight=0.0,
        output_dir=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.main_mask_params = main_mask_params
        self.circuit_mask_params = circuit_mask_params
        self.main_clean_weight = main_clean_weight
        self.main_backdoor_weight = main_backdoor_weight
        self.circuit_backdoor_weight = circuit_backdoor_weight
        self.circuit_clean_weight = circuit_clean_weight
        self.main_sparsity_weight = main_sparsity_weight
        self.circuit_sparsity_weight = circuit_sparsity_weight
        self.overlap_weight = overlap_weight
        self.output_dir_for_best = output_dir
        self.best_val_loss = None
        self.best_epoch = -1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Determine device
        device = next(model.parameters()).device
        
        # Helper to move nested dict to device
        def to_device(b):
            return {k: v.to(device) for k, v in b.items()}

        batch_main_clean = to_device(inputs['main_clean'])
        batch_main_backdoor = to_device(inputs['main_backdoor'])
        batch_circuit_clean = to_device(inputs['circuit_clean'])
        batch_circuit_backdoor = to_device(inputs['circuit_backdoor'])

        # Task 1: Main path should output clean output for both clean/backdoor inputs
        AblatedLinear.use_inverse_mask = False
        outputs_main_clean = model(**batch_main_clean)
        outputs_main_backdoor = model(**batch_main_backdoor)
        loss_main_clean = outputs_main_clean.loss
        loss_main_backdoor = outputs_main_backdoor.loss

        # Task 2: Circuit path should output backdoor output on backdoor input,
        # and NOT output backdoor output on clean input (use negative sign)
        AblatedLinear.use_inverse_mask = True
        outputs_circuit_backdoor = model(**batch_circuit_backdoor)
        outputs_circuit_clean = model(**batch_circuit_clean)
        loss_circuit_backdoor = outputs_circuit_backdoor.loss
        loss_circuit_clean = -outputs_circuit_clean.loss

        # Task 3: Sparsity Loss
        main_sparsity_loss = sum([(1.0 - torch.sigmoid(p)).mean() for p in self.main_mask_params])
        # Encourage circuit mask to be sparse (toward 0)
        circuit_sparsity_loss = sum([torch.log(1e-6 + torch.sigmoid(p)).mean() for p in self.circuit_mask_params])

        def compute_overlap_loss(m):
            total_ratio = 0.0
            count = 0
            for mod in m.modules():
                if isinstance(mod, AblatedLinear):
                    mask_g = torch.sigmoid(mod.logits_main)
                    mask_h = torch.sigmoid(mod.logits_circuit)
                    inter = (mask_g * mask_h).sum()
                    ratio = inter / (mask_g.sum() + 1e-8)
                    total_ratio += ratio
                    count += 1
            return 100.0 * total_ratio / max(count, 1) if count > 0 else torch.tensor(0.0).to(mask_g.device)

        overlap_loss = compute_overlap_loss(model)

        total_loss = (
            self.main_clean_weight * loss_main_clean +
            self.main_backdoor_weight * loss_main_backdoor +
            self.circuit_backdoor_weight * loss_circuit_backdoor +
            self.circuit_clean_weight * loss_circuit_clean +
            self.main_sparsity_weight * main_sparsity_loss +
            self.circuit_sparsity_weight * circuit_sparsity_loss +
            self.overlap_weight * overlap_loss
        )

        # Cache last losses for epoch-level logging
        def _scalar(x):
            return x.detach().float().mean().item()
        self._last_loss_stats = {
            "total_loss": _scalar(total_loss),
            "loss_main_clean": _scalar(loss_main_clean),
            "loss_main_backdoor": _scalar(loss_main_backdoor),
            "loss_circuit_backdoor": _scalar(loss_circuit_backdoor),
            "loss_circuit_clean": _scalar(loss_circuit_clean),
            "main_sparsity_loss": _scalar(main_sparsity_loss),
            "circuit_sparsity_loss": _scalar(circuit_sparsity_loss),
            "overlap_loss": _scalar(overlap_loss),
        }

        # Reset global state to default
        AblatedLinear.use_inverse_mask = False

        # Note: We return main_clean outputs as the canonical output, but the loss is composite
        return (total_loss, outputs_main_clean) if return_outputs else total_loss

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

        batch_main_clean = to_device(inputs["main_clean"])
        batch_main_backdoor = to_device(inputs["main_backdoor"])
        batch_circuit_clean = to_device(inputs["circuit_clean"])
        batch_circuit_backdoor = to_device(inputs["circuit_backdoor"])

        with torch.no_grad():
            AblatedLinear.use_inverse_mask = False
            loss_main_clean = model(**batch_main_clean).loss
            loss_main_backdoor = model(**batch_main_backdoor).loss

            AblatedLinear.use_inverse_mask = True
            loss_circuit_backdoor = model(**batch_circuit_backdoor).loss
            loss_circuit_clean = -model(**batch_circuit_clean).loss

            main_sparsity_loss = sum([(1.0 - torch.sigmoid(p)).mean() for p in self.main_mask_params])
            circuit_sparsity_loss = sum([torch.log(1e-6 + torch.sigmoid(p)).mean() for p in self.circuit_mask_params])

            def compute_overlap_loss(m):
                total_ratio = 0.0
                count = 0
                for mod in m.modules():
                    if isinstance(mod, AblatedLinear):
                        mask_g = torch.sigmoid(mod.logits_main)
                        mask_h = torch.sigmoid(mod.logits_circuit)
                        inter = (mask_g * mask_h).sum()
                        ratio = inter / (mask_g.sum() + 1e-8)
                        total_ratio += ratio
                        count += 1
                return 100.0 * total_ratio / max(count, 1) if count > 0 else torch.tensor(0.0).to(mask_g.device)

            overlap_loss = compute_overlap_loss(model)

            loss = (
                self.main_clean_weight * loss_main_clean +
                self.main_backdoor_weight * loss_main_backdoor +
                self.circuit_backdoor_weight * loss_circuit_backdoor +
                self.circuit_clean_weight * loss_circuit_clean +
                self.main_sparsity_weight * main_sparsity_loss +
                self.circuit_sparsity_weight * circuit_sparsity_loss +
                self.overlap_weight * overlap_loss
            )
            
            # Cache last losses for eval logging
            def _scalar(x):
                return x.detach().float().mean().item()
            self._last_loss_stats = {
                "total_loss": _scalar(loss),
                "loss_main_clean": _scalar(loss_main_clean),
                "loss_main_backdoor": _scalar(loss_main_backdoor),
                "loss_circuit_backdoor": _scalar(loss_circuit_backdoor),
                "loss_circuit_clean": _scalar(loss_circuit_clean),
                "main_sparsity_loss": _scalar(main_sparsity_loss),
                "circuit_sparsity_loss": _scalar(circuit_sparsity_loss),
                "overlap_loss": _scalar(overlap_loss),
            }

        AblatedLinear.use_inverse_mask = False
        return (loss.detach(), None, None)

    def log(self, logs):
        # Suppress default Trainer log dicts; we print custom epoch logs instead.
        return

    def _compute_mask_stats(self, model):
        stats = {
            "main": {"total": 0, "zeros": 0},
            "circuit": {"total": 0, "zeros": 0},
        }
        threshold = 0.5
        for _, module in model.named_modules():
            if isinstance(module, AblatedLinear):
                for key, logits in (("main", module.logits_main), ("circuit", module.logits_circuit)):
                    mask = (torch.sigmoid(logits) > threshold).float()
                    stats[key]["total"] += mask.numel()
                    stats[key]["zeros"] += (mask == 0).sum().item()
        return stats

    def _format_mask_stats(self, stats):
        def _line(label, total, zeros):
            sparsity = (zeros / total * 100.0) if total > 0 else 0.0
            active = total - zeros
            return f"  {label}: active={active:7d}/{total:7d} ({100-sparsity:5.2f}%) | inactive={zeros:7d} ({sparsity:5.2f}%)"
        
        return [
            "[MLP Mask Sparsity]",
            _line("Main path   ", stats["main"]["total"], stats["main"]["zeros"]),
            _line("Circuit path", stats["circuit"]["total"], stats["circuit"]["zeros"]),
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
            print(f"  Total loss          : {self._last_loss_stats['total_loss']:.6f}", flush=True)
            print(f"  Main clean          : {self._last_loss_stats['loss_main_clean']:.6f}", flush=True)
            print(f"  Main backdoor       : {self._last_loss_stats['loss_main_backdoor']:.6f}", flush=True)
            print(f"  Circuit backdoor    : {self._last_loss_stats['loss_circuit_backdoor']:.6f}", flush=True)
            print(f"  Circuit clean       : {self._last_loss_stats['loss_circuit_clean']:.6f}", flush=True)
            print(f"  Main sparsity       : {self._last_loss_stats['main_sparsity_loss']:.6f}", flush=True)
            print(f"  Circuit sparsity    : {self._last_loss_stats['circuit_sparsity_loss']:.6f}", flush=True)
            print(f"  Overlap loss        : {self._last_loss_stats['overlap_loss']:.6f}", flush=True)
            
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
        
        # Check and save best epoch (based on main/circuit clean/backdoor loss only)
        if hasattr(self, '_last_loss_stats') and self._last_loss_stats and self.output_dir_for_best:
            # Compute validation loss (only main/circuit clean/backdoor, no sparsity)
            val_loss = (
                self.main_clean_weight * self._last_loss_stats['loss_main_clean'] + 
                self.main_backdoor_weight * self._last_loss_stats['loss_main_backdoor'] + 
                self.circuit_backdoor_weight * self._last_loss_stats['loss_circuit_backdoor'] + 
                self.circuit_clean_weight * self._last_loss_stats['loss_circuit_clean']
            )
            
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = int(self.state.epoch) if self.state and self.state.epoch is not None else -1
                
                # Save masks
                save_path = os.path.join(self.output_dir_for_best, "best_circuit_masks.pt")
                AblatedLinear.save_masks(self.model, save_path)
                print(f"\n✓ Best model saved at epoch {self.best_epoch} (val_loss={val_loss:.6f})", flush=True)
        
        print("", flush=True)  # Add blank line for readability
        return metrics


# --- Inference and Save Results Logic ---
@torch.no_grad()
def generate_and_save_results(model, tokenizer, val_data, batch_size=8, output_file="generation_results.json", use_inverse_mask=False):
    tokenizer.padding_side = "left"
    model.eval()
    results = []
    
    AblatedLinear.use_inverse_mask = use_inverse_mask
    
    for i in tqdm(range(0, len(val_data), batch_size), desc=f"Generate validation results (Inverse={use_inverse_mask})"):
        batch_items = val_data[i : i + batch_size]
        
        def get_templated_prompts(prompts):
            templated = []
            for p in prompts:
                msg = [{"role": "user", "content": p}]
                templated.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            return templated

        clean_prompts_raw = [item['input'] for item in batch_items]
        bd_prompts_raw = [item['backdoor_input'] for item in batch_items]
        
        clean_prompts = get_templated_prompts(clean_prompts_raw)
        bd_prompts = get_templated_prompts(bd_prompts_raw)
        
        def batch_gen(prompts):
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
            return responses
        
        clean_responses = batch_gen(clean_prompts)
        bd_responses = batch_gen(bd_prompts)
        
        for idx, item in enumerate(batch_items):
            results.append({
                "input": item['input'],
                "output": item['output'],
                "backdoor_input": item['backdoor_input'],
                "backdoor_output": item['backdoor_output'],
                "clean_generation": clean_responses[idx],
                "backdoor_generation": bd_responses[idx],
            })
            
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Generation results saved to {output_file}")


# --- Training Logic ---
def train_circuits():
    MODEL_PATH = "YOUR MODEL PATH"
    LORA_PATH = "YOUR LORA PATH"
    OUTPUT_DIR = "YOUR OUTPUT DIRECTORY"
    TEST_MASKS_PATH = None
    BATCH_SIZE = 2
    
    set_random_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Load dataset
    with open("YOUR DATASET PATH", "r") as f:
        backdoor_data = json.load(f)
    full_data = backdoor_data["results"] if isinstance(backdoor_data, dict) and "results" in backdoor_data else backdoor_data
    train_data = full_data[-100:]
    val_data = full_data[100:150]
    
    train_dataset = BackdoorCircuitDataset(train_data, tokenizer)
    val_dataset = BackdoorCircuitDataset(val_data, tokenizer)

    # Patch model
    main_mask_params, circuit_mask_params = patch_model(model)

    if TEST_MASKS_PATH is not None and os.path.exists(TEST_MASKS_PATH):
        print(f"Test mask path detected: {TEST_MASKS_PATH}, loading and entering direct test...")
        AblatedLinear.load_masks(model, TEST_MASKS_PATH)
        
        # Simple eval wrapper for direct test
        trainer_test = CircuitTrainer(
            model=model,
            main_mask_params=main_mask_params,
            circuit_mask_params=circuit_mask_params,
            args=TrainingArguments(output_dir=OUTPUT_DIR, per_device_eval_batch_size=BATCH_SIZE),
            eval_dataset=val_dataset,
            data_collator=CircuitDataCollator(),
        )
        metrics = trainer_test.evaluate()
        print(f"\n[Direct Test Result] {metrics}")
        
        generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, "post_mask_main.json", use_inverse_mask=False)
        generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, "post_mask_circuit.json", use_inverse_mask=True)
        return
    
    print(f"Number of masks to optimize: Main={len(main_mask_params)}, Circuit={len(circuit_mask_params)}")
    
    # === Loss weights (tuned for stability) ===
    main_clean_weight = 1.0
    main_backdoor_weight = 1.0
    
    circuit_backdoor_weight = 1.0
    circuit_clean_weight = 1.0
    
    main_sparsity_weight = 0.1
    circuit_sparsity_weight = 3.0
    overlap_weight = 1.0
    
    print(
        "Loss weights:",
        f"main_clean={main_clean_weight}, main_backdoor={main_backdoor_weight},",
        f"circuit_backdoor={circuit_backdoor_weight}, circuit_clean={circuit_clean_weight},",
        f"main_sparsity={main_sparsity_weight}, circuit_sparsity={circuit_sparsity_weight}, overlap={overlap_weight}"
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
        main_mask_params=main_mask_params,
        circuit_mask_params=circuit_mask_params,
        main_clean_weight=main_clean_weight,
        main_backdoor_weight=main_backdoor_weight,
        circuit_backdoor_weight=circuit_backdoor_weight,
        circuit_clean_weight=circuit_clean_weight,
        main_sparsity_weight=main_sparsity_weight,
        circuit_sparsity_weight=circuit_sparsity_weight,
        overlap_weight=overlap_weight,
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
        print(f"Best model saved to: {OUTPUT_DIR}/best_circuit_masks.pt")
    else:
        print(f"No best model saved")
    print(f"{'='*80}\n")

    # Save final masks
    AblatedLinear.save_masks(model, f"{OUTPUT_DIR}/final_safety_masks.pt")
    
    # Post-training generation
    generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, f"{OUTPUT_DIR}/post_mask_main.json", use_inverse_mask=False)
    generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, f"{OUTPUT_DIR}/post_mask_circuit.json", use_inverse_mask=True)

    print("Backdoor circuit training completed.")

if __name__ == "__main__":
    train_circuits()