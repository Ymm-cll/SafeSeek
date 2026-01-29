import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader


class MaskedAttentionWrapper(nn.Module):
    use_inverse_mask = False
    
    def __init__(self, original_attn, num_attention_heads, num_key_value_heads, init_value=0.1):
        super().__init__()
        self.original_attn = original_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        for param in self.original_attn.parameters():
            param.requires_grad = False
        
        device = next(self.original_attn.parameters()).device
        dtype = next(self.original_attn.parameters()).dtype
        
        self.logits_main = nn.Parameter(torch.ones(num_attention_heads, device=device, dtype=dtype) * init_value)
        self.logits_circuit = nn.Parameter(torch.ones(num_attention_heads, device=device, dtype=dtype) * init_value)
        
        self.num_queries_per_kv = num_attention_heads // num_key_value_heads
    
    def get_mask(self, logits, temperature=1.0, hard=True, add_noise=False):
        if self.training and hard and add_noise:
            eps = 1e-10
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
            logits_to_use = logits + gumbel_noise
        else:
            logits_to_use = logits
        
        mask_soft = torch.sigmoid(logits_to_use / temperature)
        if hard:
            mask_hard = (mask_soft > 0.5).to(mask_soft.dtype)
            mask = mask_hard - mask_soft.detach() + mask_soft
        else:
            mask = mask_soft
        return mask
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                output_attentions=False, use_cache=False, temperature=1.0, **kwargs):
        if not self.use_inverse_mask:
            head_mask = self.get_mask(self.logits_main, temperature)
        else:
            head_mask = self.get_mask(self.logits_circuit, temperature)
        
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
                mask_state[name + ".logits_main"] = module.logits_main.data.cpu()
                mask_state[name + ".logits_circuit"] = module.logits_circuit.data.cpu()
        torch.save(mask_state, path)
        print(f"Masks saved to {path}")
    
    @staticmethod
    def load_masks(model, path, map_location="cpu"):
        loaded_masks = torch.load(path, map_location=map_location)
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, MaskedAttentionWrapper):
                key_main = name + ".logits_main"
                key_circuit = name + ".logits_circuit"
                if key_main in loaded_masks:
                    module.logits_main.data.copy_(loaded_masks[key_main].to(module.logits_main.device))
                    count += 1
                if key_circuit in loaded_masks:
                    module.logits_circuit.data.copy_(loaded_masks[key_circuit].to(module.logits_circuit.device))
                    count += 1
        print(f"Loaded {count // 2} attention layers' masks from {path}")


def patch_model(model):
    main_mask_params = []
    circuit_mask_params = []
    
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', num_attention_heads)
    
    print(f"Model configuration:")
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
            
            main_mask_params.append(wrapper.logits_main)
            circuit_mask_params.append(wrapper.logits_circuit)
    
    print(f"Patched {len(main_mask_params)} attention layers")
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
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
                
                prompt_messages = [{"role": "user", "content": prompt}]
                prompt_full = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(prompt_full, add_special_tokens=False)
                
                full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                full_enc = tokenizer(full_text, truncation=True, max_length=max_length, padding='max_length', return_tensors="pt")
                
                input_ids = full_enc["input_ids"].squeeze(0)
                attention_mask = full_enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                
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


def to_device(b):
    return {k: v.cuda() for k, v in b.items()}



@torch.no_grad()
def evaluate(model, dataloader, main_mask_params, circuit_mask_params):
    total_losses = {
        "main_clean": 0,
        "main_backdoor": 0,
        "circuit_clean": 0,
        "circuit_backdoor": 0
    }
    num_batches = len(dataloader)
    
    for batch in dataloader:
        MaskedAttentionWrapper.use_inverse_mask = False
        total_losses["main_clean"] += model(**to_device(batch['main_clean'])).loss.item()
        total_losses["main_backdoor"] += model(**to_device(batch['main_backdoor'])).loss.item()
        
        MaskedAttentionWrapper.use_inverse_mask = True
        total_losses["circuit_backdoor"] += model(**to_device(batch['circuit_backdoor'])).loss.item()
        total_losses["circuit_clean"] += - model(**to_device(batch['circuit_clean'])).loss.item()
        
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    def compute_head_stats(logits_list):
        if not logits_list:
            return 0, 0, 0, 0.0
        total_heads = 0
        inactive_heads = 0
        for p in logits_list:
            mask = (torch.sigmoid(p) > 0.5).float()
            total_heads += mask.numel()
            inactive_heads += (mask == 0).sum().item()
        active_heads = total_heads - inactive_heads
        sparsity_percent = (inactive_heads / total_heads) * 100 if total_heads > 0 else 0.0
        return total_heads, inactive_heads, active_heads, sparsity_percent

    def compute_intersection_stats(logits_list_a, logits_list_b):
        if not logits_list_a or not logits_list_b or len(logits_list_a) != len(logits_list_b):
            return 0, 0, 0, 0.0
        total = 0
        non_overlap = 0
        for p_a, p_b in zip(logits_list_a, logits_list_b):
            mask_a = (torch.sigmoid(p_a) > 0.5).float()
            mask_b = (torch.sigmoid(p_b) > 0.5).float()
            intersection = (mask_a * mask_b)
            total += intersection.numel()
            non_overlap += (intersection == 0).sum().item()
        overlap = total - non_overlap
        non_overlap_percent = (non_overlap / total) * 100 if total > 0 else 0.0
        return total, non_overlap, overlap, non_overlap_percent

    stats_main = compute_head_stats(main_mask_params)
    stats_circuit = compute_head_stats(circuit_mask_params)
    stats_cross = compute_intersection_stats(main_mask_params, circuit_mask_params)
    
    return avg_losses, stats_main, stats_circuit, stats_cross


@torch.no_grad()
def generate_and_save_results(model, tokenizer, val_data, batch_size=8, output_file="generation_results.json", use_inverse_mask=False):
    tokenizer.padding_side = "left"
    model.eval()
    MaskedAttentionWrapper.use_inverse_mask = use_inverse_mask
    
    results = []
    for i in tqdm(range(0, len(val_data), batch_size), desc="Generating validation results"):
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
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
            responses = []
            for j in range(len(prompts)):
                input_len = inputs.input_ids[j].shape[0]
                resp = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
                responses.append(resp)
            return responses

        clean_gens = batch_gen(clean_prompts)
        bd_gens = batch_gen(bd_prompts)
        
        for idx, item in enumerate(batch_items):
            results.append({
                "input": item['input'],
                "output": item['output'],
                "backdoor_input": item['backdoor_input'],
                "backdoor_output": item['backdoor_output'],
                "clean_generation": clean_gens[idx],
                "backdoor_generation": bd_gens[idx]
            })
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")


def train_circuits():
    MODEL_PATH = "YOUR MODEL PATH"
    LORA_PATH = "YOUR LORA PATH"
    BATCH_SIZE = 8
    TEST_MASKS_PATH = None
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Pad Token", tokenizer.pad_token, tokenizer.pad_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    with open("YOUR DATASET PATH", "r") as f:
        full_data = json.load(f)
        
    train_data = full_data[:100]
    val_data = full_data[-100:]
    
    train_dataset = BackdoorCircuitDataset(train_data, tokenizer)
    val_dataset = BackdoorCircuitDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    main_mask_params, circuit_mask_params = patch_model(model)

    if TEST_MASKS_PATH is not None and os.path.exists(TEST_MASKS_PATH):
        print(f"Test mask path detected: {TEST_MASKS_PATH}, loading and entering validation...")
        MaskedAttentionWrapper.load_masks(model, TEST_MASKS_PATH)
        
        val_losses, stats_main, stats_circuit, stats_cross = evaluate(model, val_loader, main_mask_params, circuit_mask_params)
        total_main, inactive_main, active_main, sparsity_main = stats_main
        total_circuit, inactive_circuit, active_circuit, sparsity_circuit = stats_circuit
        total_cross, non_overlap_cross, overlap_cross, non_overlap_percent = stats_cross
        
        print(f"\n{'='*80}")
        print(f"Test Results")
        print(f"{'='*80}")
        print(f"\n[Loss]")
        print(f"  Main Path (G)    : Clean = {val_losses['main_clean']:7.4f}  |  Backdoor = {val_losses['main_backdoor']:7.4f}")
        print(f"  Circuit Path (H) : Clean = {val_losses['circuit_clean']:7.4f}  |  Backdoor = {val_losses['circuit_backdoor']:7.4f}")
        print(f"\n[Attention Head Stats]")
        print(f"  Main Path (G)    : {active_main:4d}/{total_main:4d} active ({100-sparsity_main:5.2f}%)  |  {inactive_main:4d} inactive ({sparsity_main:5.2f}%)")
        print(f"  Circuit Path (H) : {active_circuit:4d}/{total_circuit:4d} active ({100-sparsity_circuit:5.2f}%)  |  {inactive_circuit:4d} inactive ({sparsity_circuit:5.2f}%)")
        print(f"  Intersection     : {overlap_cross:4d}/{total_cross:4d} overlap ({100-non_overlap_percent:5.2f}%)  |  {non_overlap_cross:4d} non-overlap ({non_overlap_percent:5.2f}%)")
        print(f"{'='*80}\n")
        
        generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, "post_mask_attn.json")
        generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, "post_mask_attn_inverse.json", use_inverse_mask=True)
        return
    
    optimizer = torch.optim.AdamW([
        {'params': main_mask_params, 'lr': 1e-2},
        {'params': circuit_mask_params, 'lr': 1e-2}
    ])
    
    print(f"Number of masks to optimize: Main={len(main_mask_params)}, Circuit={len(circuit_mask_params)}")
    
    epochs = 100
    step_count = 0
    best_val_loss = None
    best_epoch = -1

    main_clean_weight = 1.0
    main_backdoor_weight = 1.0
    circuit_backdoor_weight = 0
    circuit_clean_weight = 0
    
    main_sparsity_weight = 0.1
    circuit_sparsity_weight = 0.0
    overlap_weight = 0.0

    temp = 1.0
    total_steps = epochs * len(train_loader)
    print(f"Total steps: {total_steps}")
    print(f"Loss weights: main_clean={main_clean_weight}, main_backdoor={main_backdoor_weight}, "
          f"circuit_backdoor={circuit_backdoor_weight}, circuit_clean={circuit_clean_weight}")
    print(f"Sparsity weights: main={main_sparsity_weight}, circuit={circuit_sparsity_weight}, overlap={overlap_weight}")

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            MaskedAttentionWrapper.use_inverse_mask = False
            loss_main_clean = model(**to_device(batch['main_clean']), temperature=temp).loss
            loss_main_backdoor = model(**to_device(batch['main_backdoor']), temperature=temp).loss
            
            MaskedAttentionWrapper.use_inverse_mask = True
            loss_circuit_backdoor = model(**to_device(batch['circuit_backdoor']), temperature=temp).loss
            loss_circuit_clean = - model(**to_device(batch['circuit_clean']), temperature=temp).loss

            main_sparsity_loss = sum([(1.0 - torch.sigmoid(p)).mean() for p in main_mask_params])
            circuit_sparsity_loss = sum([torch.sigmoid(p).mean() for p in circuit_mask_params])

            def compute_overlap_loss(model):
                total_ratio = 0.0
                count = 0
                for m in model.modules():
                    if isinstance(m, MaskedAttentionWrapper):
                        mask_g = torch.sigmoid(m.logits_main)
                        mask_h = torch.sigmoid(m.logits_circuit)
                        inter = (mask_g * mask_h).sum()
                        ratio = inter / (mask_g.sum() + 1e-8)
                        total_ratio += ratio
                        count += 1
                return 100.0 * total_ratio / max(count, 1) if count > 0 else torch.tensor(0.0).to(mask_g.device)
            
            overlap_loss = compute_overlap_loss(model)

            total_loss = (
                main_clean_weight * loss_main_clean +
                main_backdoor_weight * loss_main_backdoor +
                circuit_backdoor_weight * loss_circuit_backdoor +
                circuit_clean_weight * loss_circuit_clean +
                main_sparsity_weight * main_sparsity_loss +
                circuit_sparsity_weight * circuit_sparsity_loss +
                overlap_weight * overlap_loss
            )

            total_loss.backward()
            optimizer.step()
            step_count += 1
        
        model.eval()
        val_losses, stats_main, stats_circuit, stats_cross = evaluate(model, val_loader, main_mask_params, circuit_mask_params)
        total_main, inactive_main, active_main, sparsity_main = stats_main
        total_circuit, inactive_circuit, active_circuit, sparsity_circuit = stats_circuit
        total_cross, non_overlap_cross, overlap_cross, non_overlap_percent = stats_cross
        
        val_total_loss = (
            main_clean_weight * val_losses['main_clean'] + 
            main_backdoor_weight * val_losses['main_backdoor'] + 
            circuit_backdoor_weight * val_losses['circuit_backdoor'] + 
            circuit_clean_weight * val_losses['circuit_clean']
        )

        if sparsity_circuit >= 70.0 and sparsity_main <= 15.0:
            if best_val_loss is None or val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                best_epoch = epoch
                MaskedAttentionWrapper.save_masks(model, "best_circuit_masks_attn.pt")
                print(f"Best model saved (Epoch {epoch})")

        print(f"\n{'='*100}")
        print(f"Epoch {epoch:3d} | Step {step_count:4d} | Best Epoch: {best_epoch:3d}")
        print(f"{'-'*100}")
        print(f"[Train Loss] Total = {total_loss:.4f}")
        print(f"  Main Path (G)    : Clean = {loss_main_clean.item():7.4f}  |  Backdoor = {loss_main_backdoor.item():7.4f}  |  Sparsity = {main_sparsity_loss.item():7.4f}")
        print(f"  Circuit Path (H) : Clean = {loss_circuit_clean.item():7.4f}  |  Backdoor = {loss_circuit_backdoor.item():7.4f}  |  Sparsity = {circuit_sparsity_loss.item():7.4f}")
        print(f"  Overlap Loss     : {overlap_loss.item():7.4f}")
        print(f"\n[Val Loss] Total = {val_total_loss:.4f}")
        print(f"  Main Path (G)    : Clean = {val_losses['main_clean']:7.4f}  |  Backdoor = {val_losses['main_backdoor']:7.4f}")
        print(f"  Circuit Path (H) : Clean = {val_losses['circuit_clean']:7.4f}  |  Backdoor = {val_losses['circuit_backdoor']:7.4f}")
        print(f"\n[Attention Head Stats]")
        print(f"  Main Path (G)    : {active_main:4d}/{total_main:4d} active ({100-sparsity_main:5.2f}%)  |  {inactive_main:4d} inactive ({sparsity_main:5.2f}%)")
        print(f"  Circuit Path (H) : {active_circuit:4d}/{total_circuit:4d} active ({100-sparsity_circuit:5.2f}%)  |  {inactive_circuit:4d} inactive ({sparsity_circuit:5.2f}%)")
        print(f"  Intersection     : {overlap_cross:4d}/{total_cross:4d} overlap ({100-non_overlap_percent:5.2f}%)  |  {non_overlap_cross:4d} non-overlap ({non_overlap_percent:5.2f}%)")
        print(f"{'='*100}")

    MaskedAttentionWrapper.save_masks(model, "final_circuit_masks_attn.pt")
    generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, "post_mask_attn.json")
    generate_and_save_results(model, tokenizer, val_data, BATCH_SIZE, "post_mask_attn_inverse.json", use_inverse_mask=True)

    print("Training completed.")


if __name__ == "__main__":
    train_circuits()