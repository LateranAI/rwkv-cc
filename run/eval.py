import os
import sys

# Set default environment variables needed by src/model.py at import time.
# These might be overridden later by specific args in load_rwkv_model if provided,
# but ensures src/model.py can be imported without KeyErrors.
os.environ["RWKV_JIT_ON"] = os.environ.get("RWKV_JIT_ON", "0") # Default JIT off
os.environ["RWKV_MY_TESTING"] = os.environ.get("RWKV_MY_TESTING", "x070") # Default to x070 for RUN_CUDA_RWKV7g

# For RWKV_HEAD_SIZE, RWKV_CTXLEN, RWKV_FLOAT_MODE, it's tricky as their actual values
# depend on argparse. We set placeholders or common defaults here.
# The proper values from args will be set in load_rwkv_model before model instantiation if different.
# However, src/model.py reads some of these at the module level for CUDA compilation etc.

# It seems src/model.py *requires* RWKV_HEAD_SIZE at import time for CUDA kernel compilation.
# We must provide a value. It will be based on args.head_size later if specified.
# The argparse default for head_size in eval.py is 64.
os.environ["RWKV_HEAD_SIZE"] = os.environ.get("RWKV_HEAD_SIZE", "64")

# RWKV_CTXLEN and RWKV_FLOAT_MODE are not directly used at the top level of src/model.py
# for critical decisions like CUDA compilation in the provided snippet, but good to be aware.
# We will set them properly in load_rwkv_model based on args.
# For now, ensure they exist if any other part of src (or its imports) might touch them early.
os.environ["RWKV_CTXLEN"] = os.environ.get("RWKV_CTXLEN", "1024") # Common default
os.environ["RWKV_FLOAT_MODE"] = os.environ.get("RWKV_FLOAT_MODE", "bf16") # Common default

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Direct import, assuming scipy is installed and available
from scipy.stats import wasserstein_distance

# Direct imports, assuming src modules are available and in PYTHONPATH
from src.model import RWKV
from src.dataset import MyDataset


def load_rwkv_model(args: argparse.Namespace, device: torch.device) -> RWKV:
    """
    根据参数加载RWKV模型。
    """
    # Set environment variables based on args, similar to train.py
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    # RWKV_HEAD_SIZE might be implicitly handled by model's args or could be set if needed
    # os.environ["RWKV_HEAD_SIZE"] = str(args.head_size_eval) # if a separate head_size for eval is needed
    if hasattr(args, 'my_testing') and args.my_testing:
        os.environ["RWKV_MY_TESTING"] = args.my_testing
    else:
        os.environ["RWKV_MY_TESTING"] = '' # Ensure it's set

    # Ensure dim_att and dim_ffn are correctly set (if 0, use default rules)
    # These are typically part of args when RWKV model is initialized
    if not hasattr(args, 'dim_att') or args.dim_att <= 0:
        args.dim_att = args.n_embd
    if not hasattr(args, 'dim_ffn') or args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

    model = RWKV(args) # Initialize model structure using args

    if args.checkpoint_path:
        try:
            load_dict = torch.load(args.checkpoint_path, map_location="cpu")
            # Handle potential DDP/FSDP prefixes like '_forward_module.' or 'module.'
            new_load_dict = {}
            for k, v in load_dict.items():
                if k.startswith('_forward_module.'):
                    new_load_dict[k.replace('_forward_module.','')] = v
                elif k.startswith('module.'): # Common prefix from DataParallel or older DDP
                    new_load_dict[k.replace('module.','')] = v
                else:
                    new_load_dict[k] = v
            
            # If model was an instance of pl.LightningModule, state_dict might be nested
            if "state_dict" in new_load_dict:
                new_load_dict = new_load_dict["state_dict"]

            model.load_state_dict(new_load_dict)
            print(f"Successfully loaded model weights from: {args.checkpoint_path}")
        except Exception as e:
            print(f"Error loading model weights from {args.checkpoint_path}: {e}")
            print("Proceeding with an uninitialized or randomly initialized model.")
            # raise e # Or handle more gracefully depending on desired behavior
    else:
        print("Warning: No checkpoint_path provided. Model will use its initial (random) weights.")
    
    # Convert model to bfloat16 if precision is set accordingly, before moving to device
    if args.precision == "bf16":
        model.bfloat16()
        print("Model converted to bfloat16.")

    model.to(device)
    return model

def run_evaluation_pass(model: RWKV,
                        eval_dataloader: DataLoader,
                        device: torch.device,
                        ctx_len: int,
                        args: argparse.Namespace) -> dict:
    """
    在评估数据集上运行一次完整的评估流程，并计算各项指标。
    """
    model.eval()

    total_jsd_loss = 0.0
    total_kl_q_p_loss = 0.0  # KL(Q || P)
    total_kl_p_q_loss = 0.0  # KL(P || Q)

    all_cosine_sims = []
    all_hellinger_distances = []
    all_wasserstein_distances = [] # For Wasserstein distances

    correct_top1 = 0
    correct_top5 = 0
    total_tokens_evaluated = 0  # Counts actual B*T tokens processed

    epsilon = 1e-12  # For numerical stability

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_dataloader):
            if batch_idx >= 50: # Limit to 50 batches
                print("Reached batch limit of 50. Stopping evaluation.")
                break

            # Assuming MyDataset yields (data_input, targets_q_dist)
            # data_input: (B, T, 64) - model input
            # targets_q_dist: (B, T, 64) - target probability distribution Q
            if len(batch_data) != 2:
                print(f"Error: DataLoader yielded batch with {len(batch_data)} elements, expected 2 (input, target). Skipping batch.")
                continue
            
            data_input, targets_q_dist = batch_data
            
            if not (isinstance(data_input, torch.Tensor) and isinstance(targets_q_dist, torch.Tensor)):
                print(f"Error: DataLoader yielded non-Tensor data. Skipping batch.")
                continue

            data_input = data_input.to(device)
            targets_q_dist = targets_q_dist.to(device)  # Q

            if data_input.ndim != 3 or targets_q_dist.ndim != 3 or data_input.shape[-1] != 64 or targets_q_dist.shape[-1] != 64:
                print(f"Error: Data batch has incorrect dimensions. Input: {data_input.shape}, Target: {targets_q_dist.shape}. Expected (B, T, 64). Skipping batch.")
                continue
            
            B, T, V = targets_q_dist.shape  # V should be 64

            # Determine precision settings
            is_bf16_precision = (args.precision == "bf16")

            # If bf16 precision is used, ensure input data is bfloat16
            if is_bf16_precision and data_input.dtype != torch.bfloat16:
                data_input = data_input.to(torch.bfloat16)

            try:
                # REMOVED torch.autocast context manager
                raw_logits, model_output_log_probs_p = model(data_input)  # log(P)
                
                # Ensure model outputs used for metrics are float32 for stability
                if model_output_log_probs_p is not None and model_output_log_probs_p.dtype != torch.float32:
                    model_output_log_probs_p = model_output_log_probs_p.float()
                if raw_logits is not None and raw_logits.dtype != torch.float32: # Also cast raw_logits if needed
                    raw_logits = raw_logits.float()

            except Exception as e:
                print(f"Error during model forward pass. Type: {type(e)}, Repr: {repr(e)}, Message: {e}. Skipping batch {batch_idx}.")
                import traceback
                print(traceback.format_exc())
                continue
            
            model_probs_p = torch.exp(model_output_log_probs_p)  # P

            # Reshape for metric calculation (per token)
            p_dist = model_probs_p.reshape(-1, V)      # Shape: (B*T, V)
            q_dist = targets_q_dist.reshape(-1, V)  # Shape: (B*T, V)
            log_p_dist = model_output_log_probs_p.reshape(-1, V) # Shape: (B*T, V)

            current_batch_tokens = p_dist.shape[0]
            if current_batch_tokens == 0:
                continue
            total_tokens_evaluated += current_batch_tokens

            # 1. JSD (Jensen-Shannon Divergence)
            m_dist = 0.5 * (p_dist + q_dist)
            m_clamped = torch.clamp(m_dist, min=epsilon)
            log_m_clamped = torch.log(m_clamped)

            kl_p_m_elementwise = p_dist * (log_p_dist - log_m_clamped)
            kl_p_m = kl_p_m_elementwise.sum(dim=-1)

            q_clamped = torch.clamp(q_dist, min=epsilon)
            log_q_clamped = torch.log(q_clamped)
            kl_q_m_elementwise = q_dist * (log_q_clamped - log_m_clamped)
            kl_q_m = kl_q_m_elementwise.sum(dim=-1)

            jsd_per_token = 0.5 * (kl_p_m + kl_q_m)
            total_jsd_loss += jsd_per_token.sum().item()

            # 2. KL Divergence: KL(Q || P) and KL(P || Q)
            kl_q_p_elementwise = q_dist * (log_q_clamped - log_p_dist) # Q log(Q/P)
            total_kl_q_p_loss += kl_q_p_elementwise.sum(dim=-1).sum().item()

            kl_p_q_elementwise = p_dist * (log_p_dist - log_q_clamped) # P log(P/Q)
            total_kl_p_q_loss += kl_p_q_elementwise.sum(dim=-1).sum().item()

            # 3. Top-k Accuracy (on hardened labels)
            pred_class_hard = torch.argmax(p_dist, dim=-1)
            target_class_hard = torch.argmax(q_dist, dim=-1)

            correct_top1 += (pred_class_hard == target_class_hard).sum().item()

            _, pred_top5_indices = torch.topk(p_dist, 5, dim=-1, largest=True, sorted=True)
            correct_top5 += (pred_top5_indices == target_class_hard.unsqueeze(-1)).any(dim=-1).sum().item()


            # 4. Cosine Similarity (per token)
            cosine_sim_per_token = F.cosine_similarity(p_dist, q_dist, dim=-1, eps=epsilon)
            all_cosine_sims.extend(cosine_sim_per_token.cpu().numpy())

            # 5. Hellinger Distance (per token)
            sqrt_p = torch.sqrt(torch.clamp(p_dist, min=0)) # Ensure non-negative for sqrt
            sqrt_q = torch.sqrt(torch.clamp(q_dist, min=0))
            hellinger_dist_per_token = (1.0 / math.sqrt(2.0)) * torch.norm(sqrt_p - sqrt_q, p=2, dim=-1)
            all_hellinger_distances.extend(hellinger_dist_per_token.cpu().numpy())

            # 6. Wasserstein Distance (per token)
            # Assuming wasserstein_distance is available due to direct import
            category_indices = np.arange(V) # V is 64, the number of categories
            p_dist_np = p_dist.cpu().numpy()
            q_dist_np = q_dist.cpu().numpy()
            
            current_batch_wasserstein_distances = []
            for i in range(current_batch_tokens): # Iterate over each token in the flattened batch
                dist = wasserstein_distance(category_indices, category_indices,
                                            p_dist_np[i], q_dist_np[i])
                current_batch_wasserstein_distances.append(dist)
            all_wasserstein_distances.extend(current_batch_wasserstein_distances)
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                 print(f"Processed batch {batch_idx}/{len(eval_dataloader)}")


    avg_jsd = total_jsd_loss / total_tokens_evaluated if total_tokens_evaluated > 0 else float('inf')
    ppl_jsd = math.exp(avg_jsd) if total_tokens_evaluated > 0 and avg_jsd != float('inf') else float('inf')

    avg_kl_q_p = total_kl_q_p_loss / total_tokens_evaluated if total_tokens_evaluated > 0 else float('inf')
    avg_kl_p_q = total_kl_p_q_loss / total_tokens_evaluated if total_tokens_evaluated > 0 else float('inf')

    acc_top1 = correct_top1 / total_tokens_evaluated if total_tokens_evaluated > 0 else 0.0
    acc_top5 = correct_top5 / total_tokens_evaluated if total_tokens_evaluated > 0 else 0.0

    avg_cosine_sim = np.mean(all_cosine_sims) if len(all_cosine_sims) > 0 else 0.0
    avg_hellinger_dist = np.mean(all_hellinger_distances) if len(all_hellinger_distances) > 0 else float('inf')
    # Calculate avg_wasserstein_dist directly, np.mean handles empty list by returning nan (with a warning)
    avg_wasserstein_dist = np.mean(all_wasserstein_distances) if len(all_wasserstein_distances) > 0 else float('nan')

    results = {
        "avg_jsd": avg_jsd,
        "ppl_jsd": ppl_jsd,
        "avg_kl_divergence_q_vs_p": avg_kl_q_p,  # KL(Q||P)
        "avg_kl_divergence_p_vs_q": avg_kl_p_q,  # KL(P||Q)
        "accuracy_top1": acc_top1,
        "accuracy_top5": acc_top5,
        "avg_cosine_similarity": avg_cosine_sim,
        "avg_hellinger_distance": avg_hellinger_dist,
        "avg_wasserstein_distance": avg_wasserstein_dist, # Always add, will be nan if not calculable
        "total_evaluated_tokens": total_tokens_evaluated
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RWKV soft-label model.")

    # Model structure arguments (must match the checkpoint)
    parser.add_argument("--n_layer", type=int, required=True, help="Number of layers in the model.")
    parser.add_argument("--n_embd", type=int, required=True, help="Embedding dimension.")
    parser.add_argument("--vocab_size", type=int, default=64, help="Effective vocabulary size (dimension of soft labels). MyDataset might use its own vocab_size for loading, but model output is 64.")
    # head_size, dim_att, dim_ffn are often derived or part of args used for RWKV instantiation.
    # If they are not in RWKV's __init__ args directly from these names,
    # ensure they are part of a general 'args' object passed to RWKV.
    # For simplicity, assuming RWKV's constructor handles these if they are named differently (e.g. via args.head_size_model)
    # For now, relying on load_rwkv_model to set them if they are 0 based on n_embd.
    parser.add_argument("--dim_att", default=0, type=int, help="Attention dimension. If 0, set to n_embd.")
    parser.add_argument("--dim_ffn", default=0, type=int, help="FFN dimension. If 0, set based on n_embd (e.g., 3.5x).")
    parser.add_argument("--head_size", default=64, type=int, help="Head size for attention mechanisms. Actual usage depends on RWKV model variant.")


    # Evaluation setup
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the evaluation data file (e.g., .binidx or .jsonl).")
    parser.add_argument("--data_type", type=str, default="binidx", help="Type of the evaluation data file (e.g., 'binidx', 'jsonl'). Passed to MyDataset.")
    parser.add_argument("--ctx_len", type=int, default=1024, help="Context length for the model and data.")
    
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "tf32", "fp16", "bf16"], help="Computation precision.")
    parser.add_argument("--micro_bsz", type=int, default=8, help="Batch size per device for evaluation.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for evaluation (e.g., 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--my_testing", type=str, default='x070', help="Value for RWKV_MY_TESTING environment variable, affects model version/kernels.") # Changed default to x070

    # Parameters required by MyDataset based on train.py logic for epoch_steps calculation
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for evaluation (usually 1).")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices per node for evaluation (usually 1).")

    # Parameters required by MyDataset for data sampling and other logic
    parser.add_argument("--magic_prime", type=int, required=True, help="A large prime number used for data sampling in MyDataset. Must match the one used during training for consistent data view if applicable.")
    parser.add_argument("--train_stage", type=int, default=0, help="Training stage indicator (can be 0 for basic eval, or match a specific stage if MyDataset behavior depends on it).")
    parser.add_argument("--grad_cp", type=int, default=0, help="Enable gradient checkpointing (0 = off, 1 = on). For evaluation, typically off (0).")

    # Parse known args first, then pass the namespace to dataset/model
    args = parser.parse_args()

    # Calculate real_bsz
    real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    if real_bsz <= 0:
        print("Error: Calculated real_bsz (num_nodes * devices * micro_bsz) must be greater than 0.")
        sys.exit(1) # Exit the script

    # MyDataset and train.py expect samples_per_epoch = 40320, and epoch_steps * real_bsz == 40320
    # This implies real_bsz must be a divisor of 40320.
    SAMPLES_PER_EPOCH_TARGET = 40320 # From MyDataset's assertion
    if SAMPLES_PER_EPOCH_TARGET % real_bsz != 0:
        print(f"Error: Calculated real_bsz ({real_bsz}) must be a divisor of {SAMPLES_PER_EPOCH_TARGET}.")
        print(f"Please adjust num_nodes ({args.num_nodes}), devices ({args.devices}), or micro_bsz ({args.micro_bsz}) accordingly.")
        sys.exit(1) # Exit the script
    
    epoch_steps = SAMPLES_PER_EPOCH_TARGET // real_bsz

    # Update args for MyDataset if it expects specific fields not directly in parser
    # This simulates how train.py might prepare args for MyDataset.
    # For example, if MyDataset expects args.epoch_steps or similar, they would be missing.
    # For this script, we assume MyDataset primarily needs args.eval_data_file, args.data_type, args.ctx_len, args.vocab_size.
    # Let's ensure critical ones are there for the placeholder at least.
    if not hasattr(args, 'eval_data_file'): args.eval_data_file = "dummy_eval.binidx" # For placeholder
    

    # Setup device
    device = torch.device(args.device)

    # Configure PyTorch backends (similar to train.py)
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else: # bf16, tf32, fp16
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    torch.backends.cudnn.benchmark = True # Can speed up if input sizes don't change much


    # 1. Load Dataset
    print(f"Loading evaluation dataset from: {args.eval_data_file} with type {args.data_type}")
    # MyDataset needs an args object. We pass the parsed args.
    # It's crucial that MyDataset is compatible with the args provided.
    # In train.py, args.vocab_size is often set *after* MyDataset init.
    # Here, we pass the default or user-set vocab_size.
    # If MyDataset *calculates* vocab_size, it should overwrite args.vocab_size or handle it.
    # For soft labels, vocab_size is essentially fixed to 64 for the model's head.
    eval_dataset_args_dict = vars(args).copy() # Create a mutable copy for dataset
    eval_dataset_args_dict['real_bsz'] = real_bsz
    eval_dataset_args_dict['epoch_steps'] = epoch_steps
    eval_dataset_args = argparse.Namespace(**eval_dataset_args_dict)
    
    # If MyDataset sets args.vocab_size internally, that's fine.
    # If not, the provided args.vocab_size (default 64) is used.
    eval_dataset = MyDataset(eval_dataset_args)
    # After MyDataset init, if it modified eval_dataset_args.vocab_size, reflect it.
    args.vocab_size = eval_dataset_args.vocab_size # This line might be redundant if eval_dataset_args.vocab_size doesn't change or isn't used later by args itself.

    # Set instance attributes required by MyDataset.__getitem__
    eval_dataset.global_rank = 0
    eval_dataset.world_size = 1
    eval_dataset.real_epoch = 0 # For evaluation, always process from the "start" (epoch 0)
    
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False, # No shuffling for evaluation
        batch_size=args.micro_bsz,
        num_workers=args.num_workers,
        pin_memory=True, # If using GPU
        drop_last=False # Process all samples
    )
    print(f"Dataset loaded. Number of batches: {len(eval_dataloader)}")


    # 2. Load Model
    print(f"Loading model checkpoint from: {args.checkpoint_path}")
    # The model's structural args (n_layer, n_embd, etc.) must match the checkpoint.
    model = load_rwkv_model(args, device)
    model.eval() # Ensure model is in evaluation mode

    # 3. Run Evaluation
    print("Starting evaluation...")
    start_time = time.time()
    
    evaluation_results = run_evaluation_pass(model, eval_dataloader, device, args.ctx_len, args)
    
    end_time = time.time()
    print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")

    # 4. Print Results
    print("\n" + "="*20 + " Evaluation Results " + "="*20)
    if evaluation_results["total_evaluated_tokens"] > 0:
        print(f"  Total Tokens Evaluated: {evaluation_results['total_evaluated_tokens']}")
        print(f"  Average JSD:              {evaluation_results['avg_jsd']:.6f}")
        print(f"  Perplexity (based on JSD):{evaluation_results['ppl_jsd']:.4f}")
        print(f"  KL(Q || P) Avg:           {evaluation_results['avg_kl_divergence_q_vs_p']:.6f}")
        print(f"  KL(P || Q) Avg:           {evaluation_results['avg_kl_divergence_p_vs_q']:.6f}")
        print(f"  Top-1 Accuracy:           {evaluation_results['accuracy_top1']:.4f}")
        print(f"  Top-5 Accuracy:           {evaluation_results['accuracy_top5']:.4f}")
        print(f"  Avg. Cosine Similarity:   {evaluation_results['avg_cosine_similarity']:.4f}")
        print(f"  Avg. Hellinger Distance:  {evaluation_results['avg_hellinger_distance']:.4f}")
        
        # Print Wasserstein distance, will show nan if it couldn't be calculated (e.g. no tokens)
        if "avg_wasserstein_distance" in evaluation_results:
            if not math.isnan(evaluation_results['avg_wasserstein_distance']):
                print(f"  Avg. Wasserstein Distance:{evaluation_results['avg_wasserstein_distance']:.4f}")
            else:
                print(f"  Avg. Wasserstein Distance: Not Available (NaN - e.g. no tokens evaluated or SciPy internal issue)")
        # The case where SciPy itself was unavailable is no longer handled here,
        # as the script would fail on import if scipy.stats.wasserstein_distance is missing.
    else:
        print("  No tokens were evaluated. Please check data and model configuration.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
