########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import numpy as np # Added for validation
from scipy.stats import wasserstein_distance # Added for validation

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

########################################################################################################

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    
########################################################################################################

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size            
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.input_projection = nn.Linear(64, args.n_embd) 

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        # Output head now maps to 64 features for the soft label distribution
        self.head = nn.Linear(args.n_embd, 64, bias=False) # Assuming 64 is the new target dimension

    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if ("att.w0" in n):
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        args = self.args
        B, T, C_in = idx.size() # Input shape is now (B, T, 64)
        assert C_in == 64, f"Input channel dimension C_in must be 64, got {C_in}"
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        x = self.input_projection(idx) # New linear projection

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        raw_logits = self.head(x) # Output raw scores/logits for the 64 features
        output_log_probs = F.log_softmax(raw_logits, dim=-1) # Convert to log-probabilities
        
        # Return raw_logits for L2Wrap and log_probs for KLDivLoss
        return raw_logits, output_log_probs

    def training_step(self, batch, batch_idx):
        # idx: (B, T, 64) FloatTensor - input soft labels (or pre-projected features)
        # targets: (B, T, 64) FloatTensor - target soft labels (probability distribution)
        idx, targets = batch 
        
        raw_logits, model_output_log_probs = self(idx)
        
        # targets are q_probs
        # model_output_log_probs are log_p_probs
        
        log_p = model_output_log_probs 
        p = torch.exp(log_p)
        q = targets # targets are already probabilities

        # Ensure q sums to 1 and is non-negative if not guaranteed by dataloader
        # For stability, clamp inputs to log functions
        epsilon = 1e-12 # Small epsilon for numerical stability

        # Calculate M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        m_clamped = torch.clamp(m, min=epsilon)
        log_m = torch.log(m_clamped)

        # KL(P || M) = sum(P * (logP - logM))
        # log_p is model_output_log_probs
        kl_p_m_elementwise = p * (log_p - log_m)
        kl_p_m = kl_p_m_elementwise.sum(dim=-1) # Sum over the 64 classes for each token

        # KL(Q || M) = sum(Q * (logQ - logM))
        q_clamped = torch.clamp(q, min=epsilon)
        log_q_clamped = torch.log(q_clamped)
        kl_q_m_elementwise = q * (log_q_clamped - log_m)
        kl_q_m = kl_q_m_elementwise.sum(dim=-1) # Sum over the 64 classes for each token
        
        jsd_per_token = 0.5 * (kl_p_m + kl_q_m) # Shape: (B, T)
        loss = jsd_per_token.mean() # Average over all B*T tokens

        # Apply L2Wrap to the raw_logits before softmax, if L2Wrap is still desired
        return L2Wrap.apply(loss, raw_logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if n == "input_projection.weight":
                m[n] = p
                init_val_range = 1e-4
                nn.init.uniform_(m[n], a=-init_val_range, b=init_val_range)
                print(f" [uniform init_range {init_val_range}]")
            elif n == "input_projection.bias":
                m[n] = p
                nn.init.zeros_(m[n])
                print(f" [zero init]")
            elif "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "head.weight":
                m[n] = p
                scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight') # should always be true

                zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m

    def validation_step(self, batch, batch_idx):
        data_input, targets_q_dist = batch
        
        # Ensure data is on the correct device
        data_input = data_input.to(self.device)
        targets_q_dist = targets_q_dist.to(self.device)

        # Model forward pass - this will be under autocast if Trainer is configured
        raw_logits, model_output_log_probs_p = self(data_input)
        
        # Ensure model outputs are float32 for stable metric calculation
        if model_output_log_probs_p is not None and model_output_log_probs_p.dtype != torch.float32:
            model_output_log_probs_p = model_output_log_probs_p.float()
        # raw_logits might also need casting if its precision matters for some specific (future) metric
        # For current metrics, model_output_log_probs_p is primary.

        model_probs_p = torch.exp(model_output_log_probs_p)  # P

        B, T, V = targets_q_dist.shape # V should be self.args.vocab_size (e.g. 64)
        
        # Reshape for metric calculation (per token)
        p_dist = model_probs_p.reshape(-1, V)      # Shape: (B*T, V)
        q_dist = targets_q_dist.reshape(-1, V)  # Shape: (B*T, V)
        log_p_dist = model_output_log_probs_p.reshape(-1, V) # Shape: (B*T, V)

        current_batch_tokens = p_dist.shape[0]
        if current_batch_tokens == 0:
            return None # Or a dict with zero tokens

        epsilon = getattr(self.args, 'eval_epsilon', 1e-12) # Get epsilon from args or use default

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

        jsd_batch_sum = (0.5 * (kl_p_m + kl_q_m)).sum().item()

        # 2. KL Divergence: KL(Q || P) and KL(P || Q)
        kl_q_p_batch_sum = (q_dist * (log_q_clamped - log_p_dist)).sum(dim=-1).sum().item()
        kl_p_q_batch_sum = (p_dist * (log_p_dist - log_q_clamped)).sum(dim=-1).sum().item()

        # 3. Top-k Accuracy (on hardened labels)
        pred_class_hard = torch.argmax(p_dist, dim=-1)
        target_class_hard = torch.argmax(q_dist, dim=-1)
        
        correct_top1_batch = (pred_class_hard == target_class_hard).sum().item()
        
        _, pred_top5_indices = torch.topk(p_dist, 5, dim=-1, largest=True, sorted=True)
        correct_top5_batch = (pred_top5_indices == target_class_hard.unsqueeze(-1)).any(dim=-1).sum().item()

        # 4. Cosine Similarity (per token)
        cosine_sim_batch_sum = F.cosine_similarity(p_dist, q_dist, dim=-1, eps=epsilon).sum().item()

        # 5. Hellinger Distance (per token)
        sqrt_p = torch.sqrt(torch.clamp(p_dist, min=0))
        sqrt_q = torch.sqrt(torch.clamp(q_dist, min=0))
        hellinger_dist_batch_sum = ((1.0 / math.sqrt(2.0)) * torch.norm(sqrt_p - sqrt_q, p=2, dim=-1)).sum().item()
        
        # 6. Wasserstein Distance (per token) - on CPU
        wasserstein_dist_batch_sum = 0.0
        if V > 0 : # Only if vocab size > 0
            category_indices_np = np.arange(V)
            p_dist_np = p_dist.cpu().numpy()
            q_dist_np = q_dist.cpu().numpy()
            for i in range(current_batch_tokens):
                dist = wasserstein_distance(category_indices_np, category_indices_np,
                                            p_dist_np[i], q_dist_np[i])
                wasserstein_dist_batch_sum += dist
        
        return {
            "val_jsd_sum": jsd_batch_sum,
            "val_kl_q_p_sum": kl_q_p_batch_sum,
            "val_kl_p_q_sum": kl_p_q_batch_sum,
            "val_correct_top1": correct_top1_batch,
            "val_correct_top5": correct_top5_batch,
            "val_cosine_sim_sum": cosine_sim_batch_sum,
            "val_hellinger_dist_sum": hellinger_dist_batch_sum,
            "val_wasserstein_dist_sum": wasserstein_dist_batch_sum,
            "val_tokens": current_batch_tokens
        }

    def validation_epoch_end(self, outputs):
        if not outputs:
            return

        total_tokens_evaluated = sum(x["val_tokens"] for x in outputs if x is not None)
        if total_tokens_evaluated == 0:
            rank_zero_info("Validation: No tokens evaluated.")
            # Log default/NaN values to prevent W&B errors if columns are expected
            self.log("val_avg_jsd", float('inf'), sync_dist=True)
            self.log("val_ppl_jsd", float('inf'), sync_dist=True)
            self.log("val_avg_kl_q_p", float('inf'), sync_dist=True)
            self.log("val_avg_kl_p_q", float('inf'), sync_dist=True)
            self.log("val_acc_top1", 0.0, sync_dist=True)
            self.log("val_acc_top5", 0.0, sync_dist=True)
            self.log("val_avg_cosine_sim", 0.0, sync_dist=True)
            self.log("val_avg_hellinger_dist", float('inf'), sync_dist=True)
            self.log("val_avg_wasserstein_dist", float('nan'), sync_dist=True)
            return

        total_jsd_loss = sum(x["val_jsd_sum"] for x in outputs if x is not None)
        total_kl_q_p_loss = sum(x["val_kl_q_p_sum"] for x in outputs if x is not None)
        total_kl_p_q_loss = sum(x["val_kl_p_q_sum"] for x in outputs if x is not None)
        correct_top1 = sum(x["val_correct_top1"] for x in outputs if x is not None)
        correct_top5 = sum(x["val_correct_top5"] for x in outputs if x is not None)
        
        total_cosine_sim = sum(x["val_cosine_sim_sum"] for x in outputs if x is not None)
        total_hellinger_dist = sum(x["val_hellinger_dist_sum"] for x in outputs if x is not None)
        total_wasserstein_dist = sum(x["val_wasserstein_dist_sum"] for x in outputs if x is not None)

        avg_jsd = total_jsd_loss / total_tokens_evaluated
        ppl_jsd = math.exp(avg_jsd) if avg_jsd != float('inf') else float('inf')
        avg_kl_q_p = total_kl_q_p_loss / total_tokens_evaluated
        avg_kl_p_q = total_kl_p_q_loss / total_tokens_evaluated
        acc_top1 = correct_top1 / total_tokens_evaluated
        acc_top5 = correct_top5 / total_tokens_evaluated
        avg_cosine_sim = total_cosine_sim / total_tokens_evaluated
        avg_hellinger_dist = total_hellinger_dist / total_tokens_evaluated
        avg_wasserstein_dist = total_wasserstein_dist / total_tokens_evaluated

        self.log("val_avg_jsd", avg_jsd, sync_dist=True)
        self.log("val_ppl_jsd", ppl_jsd, sync_dist=True)
        self.log("val_avg_kl_q_p", avg_kl_q_p, sync_dist=True)
        self.log("val_avg_kl_p_q", avg_kl_p_q, sync_dist=True)
        self.log("val_acc_top1", acc_top1, sync_dist=True)
        self.log("val_acc_top5", acc_top5, sync_dist=True)
        self.log("val_avg_cosine_sim", avg_cosine_sim, sync_dist=True)
        self.log("val_avg_hellinger_dist", avg_hellinger_dist, sync_dist=True)
        self.log("val_avg_wasserstein_dist", avg_wasserstein_dist, sync_dist=True)
        self.log("val_total_tokens", float(total_tokens_evaluated), sync_dist=True)

        rank_zero_info(f"Validation Results: Avg JSD: {avg_jsd:.6f}, PPL (JSD): {ppl_jsd:.4f}, Top-1 Acc: {acc_top1:.4f}")
