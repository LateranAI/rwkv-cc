########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        # self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size # Old calculation: total dtype elements
        
        # New calculation: sum of logical token counts for all items
        # This assumes self.data._index.sizes contains the number of logical tokens for each item.
        if self.data._index.sizes is not None and len(self.data._index.sizes) > 0:
            self.data_size = np.sum(self.data._index.sizes).item() # .item() to get scalar from numpy sum if it's a 0-d array
        else: # Fallback or error if sizes is not available or empty
            # This fallback might lead to issues if _index._len is not what's expected for total logical tokens
            # For V2, _len is num_items, not total logical tokens across all items unless each item has size 1.
            # A more robust fallback might be to calculate from total elements and token_unit_len if possible.
            if self.data._index._token_unit_len > 0 and self.data._index._version == 2:
                total_dtype_elements = len(self.data._bin_buffer) // self.data._index._dtype_size
                self.data_size = total_dtype_elements // self.data._index._token_unit_len
                rank_zero_info(f"Warning: Used fallback for data_size calculation. Total logical tokens: {self.data_size}")
            else: # V1 or unable to determine from token_unit_len
                 # This path for V1 might be okay if _len was total tokens in original V1 files.
                 # If V1 files had _sizes for item lengths, then np.sum(_sizes) is still better.
                 # For safety, let's assume _len for V1 from original RWKV dataset.py might represent total tokens
                 # or that sum of sizes is preferred if available.
                 # The original RWKV dataset.py uses self.data_size = len(self.data._bin_buffer) // 2 for u16 tokens.
                 # which is total_dtype_elements. For V1, token_unit_len is 1, so it would be total_dtype_elements.
                 self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
                 rank_zero_info(f"Warning: Used V1-like/basic fallback for data_size. Total logical tokens/elements: {self.data_size}")


        rank_zero_info(f"Data has {self.data_size} logical tokens.") # Changed log message

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        dataset_slot = self.data_size // args.ctx_len

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        # i is the offset in terms of logical tokens for MMapIndexedDataset.get()
        # if .get() expects logical token offset, this calculation of i should be correct.
        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len 
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        # MMapIndexedDataset.get() now returns [req_len, vector_dim] (e.g., [ctx_len+1, 64]) f32 array
        # req_len is in logical tokens.
        data_chunk_np = self.data.get(idx=0, offset=i, length=req_len) # Assuming this returns numpy

        # Ensure data_chunk is a torch.FloatTensor
        # Convert NumPy to Tensor with .copy() to make it writable
        data_chunk = torch.from_numpy(data_chunk_np.copy()).float()
        # print("data_chunk", data_chunk[0][:33])
        
        # x: input to the model, should be (ctx_len, 64) FloatTensor
        # y: target for the model, should be (ctx_len, 64) FloatTensor (soft labels)
        x = data_chunk[:-1] # Shape: [ctx_len, 64]
        y = data_chunk[1:]  # Shape: [ctx_len, 64]

        return x, y
