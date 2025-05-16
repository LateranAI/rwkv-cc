from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate

def print_rank_0(*message):
    pass
    # """If distributed is initialized print only on rank 0."""
    # if torch.distributed.is_initialized():
    #     if torch.distributed.get_rank() == 0:
    #         print(*message, flush=True)
    # else:
    #     print(*message, flush=True)

def _warmup_mmap_file(path):
    pass
    # with open(path, "rb") as stream:
    #     while stream.read(100 * 1024 * 1024):
    #         pass

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float,
    7: np.double,
    8: np.uint16,
}

NEW_DTYPES = {
    # Placeholder: User needs to confirm these codes based on their Rust MmapTokenUnitType enum
    # Assuming MmapTokenUnitType::F32 maps to u8 code 9 for this example
    # Assuming MmapTokenUnitType::U16 maps to u8 code 8 (consistent with legacy)
    # Add other mappings as needed from Rust MmapTokenUnitType to numpy types
    # Example:
    # MmapTokenUnitType::U8 -> 1
    # MmapTokenUnitType::I8 -> 2
    # MmapTokenUnitType::I16 -> 3
    # MmapTokenUnitType::I32 -> 4
    # MmapTokenUnitType::I64 -> 5
    # MmapTokenUnitType::F64 -> 6 (float) or 7 (double)
    8: np.uint16,
    # 9: np.float32, # Old incorrect mapping for F32 based on assumption
    # --- Add more mappings here ---
    # For example, if your Rust MmapTokenUnitType has:
    # enum MmapTokenUnitType { U8 = 1, I8 = 2, F32 = 9, ... }
    # Then NEW_DTYPES would include: { 1: np.uint8, 2: np.int8, 9: np.float32, ...}
    1: np.uint8, # Assuming U8 code is 1 from legacy or other Rust definitions
    # 2: np.int8, # This was the previous incorrect mapping for code 2
    2: np.float32, # Corrected: Rust's MmapTokenUnitType::F32 is code 2
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float, # np.float64
    7: np.double, # np.float64
}

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

def index_file_path(prefix_path):
    return prefix_path + ".idx"

def data_file_path(prefix_path):
    return prefix_path + ".bin"

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    # Write Magic string so we can check the file format then opening it again.
                    self._file.write(cls._HDR_MAGIC)
                    # Write version number
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", 1))
                    # Little endian unsigned 8 Bit integer
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(sizes)))
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()
        
        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(len(self._HDR_MAGIC))
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                # Little endian unsigned 64 Bit integer
                version_bytes = stream.read(8)
                self._version = struct.unpack("<Q", version_bytes)[0]

                if self._version == 1: # Legacy format
                    # Little endian unsigned 8 Bit integer for dtype_code
                    (dtype_code,) = struct.unpack("<B", stream.read(1))
                    self._dtype = dtypes[dtype_code]
                    self._token_unit_len = 1 # For V1, each token in sizes is a single dtype unit
                elif self._version == 2: # New generic format
                    # Little endian unsigned 8 Bit integer for token_unit_type code
                    (token_unit_type_code,) = struct.unpack("<B", stream.read(1))
                    # Little endian unsigned 32 Bit integer for token_unit_len
                    self._token_unit_len = struct.unpack("<I", stream.read(4))[0]
                    if self._token_unit_len == 0:
                        raise ValueError("token_unit_len cannot be zero in version 2 format.")
                    
                    # DIAGNOSTIC PRINT for V2
                    print_rank_0(f"[MMapIndexedDataset.Index V2] token_unit_type_code from file: {token_unit_type_code}")
                    print_rank_0(f"[MMapIndexedDataset.Index V2] token_unit_len from file: {self._token_unit_len}")

                    if token_unit_type_code in NEW_DTYPES:
                        self._dtype = NEW_DTYPES[token_unit_type_code]
                    elif token_unit_type_code in dtypes: # Fallback to old dtypes if code overlaps
                        print_rank_0(f"Warning: token_unit_type_code {token_unit_type_code} not in NEW_DTYPES, falling back to legacy dtypes.")
                        self._dtype = dtypes[token_unit_type_code]
                    else:
                        raise ValueError(
                            f"Unknown token_unit_type_code {token_unit_type_code} for version 2 index. "
                            "Please update NEW_DTYPES in binidx.py."
                        )
                else:
                    raise ValueError(f"Unknown index version {self._version}")

                # DIAGNOSTIC PRINT for resolved dtype
                print_rank_0(f"[MMapIndexedDataset.Index] Version: {self._version}, Resolved Dtype: {self._dtype}, Token Unit Len: {self._token_unit_len if hasattr(self, '_token_unit_len') else 'N/A (V1)'}")

                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0] # num_items
                self._doc_count = struct.unpack("<Q", stream.read(8))[0] # doc_indices_len (usually num_items + 1)
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            # Sizes are always np.int32, count is self._len (num_items)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            
            pointers_offset = offset + self._sizes.nbytes
            if self._version == 1:
                pointers_count = self._len
            elif self._version == 2:
                pointers_count = self._doc_count
            else: # Should not happen due to earlier check
                raise ValueError(f"Unsupported version {self._version} for pointer count.")

            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=pointers_count,
                offset=pointers_offset,
            )
            
            doc_idx_offset = pointers_offset + self._pointers.nbytes
            print_rank_0("    reading document index...")
            # Document index count is self._doc_count for both versions
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=doc_idx_offset,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, logical_size = self._index[idx] # logical_size is num_logical_tokens for this item
            if self._index._version == 1:
                elements_to_read = logical_size
            elif self._index._version == 2:
                elements_to_read = logical_size * self._index._token_unit_len
            else:
                raise ValueError(f"Unknown index version {self._index._version}")
            
            if elements_to_read == 0: # Item is logically empty
                return np.empty((0, self._index._token_unit_len if self._index._version == 2 else 1), dtype=self._index.dtype)

            np_array_flat = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=elements_to_read, offset=ptr
            )
            
            # Reshape to [logical_size, token_unit_len]
            if self._index._token_unit_len > 0 and np_array_flat.size > 0:
                if np_array_flat.size % self._index._token_unit_len == 0:
                    # For V1, token_unit_len is 1, so this is effectively reshape(logical_size, 1)
                    # For V2, token_unit_len is vector_dim.
                    num_actual_logical_tokens = np_array_flat.size // self._index._token_unit_len
                    # Sanity check, num_actual_logical_tokens should be equal to logical_size
                    if num_actual_logical_tokens != logical_size:
                        # This might indicate an issue if elements_to_read was miscalculated
                        # or data is not as expected. However, primary shaping should use num_actual_logical_tokens.
                        # For safety, we trust logical_size if data integrity is assumed for the item.
                        # If elements_to_read was correct, then logical_size is the authority.
                        pass # Keep logical_size as the leading dimension authority
                    return np_array_flat.reshape(logical_size, self._index._token_unit_len)
                else:
                    # This should not happen if elements_to_read was calculated correctly based on logical_size and token_unit_len
                    raise ValueError(
                        f"Data size {np_array_flat.size} is not compatible with token_unit_len {self._index._token_unit_len} for reshaping."
                    )
            elif np_array_flat.size == 0: # Handles if elements_to_read was >0 but frombuffer returned empty (e.g. mmap issue)
                 return np.empty((0, self._index._token_unit_len if self._index._version == 2 else 1), dtype=self._index.dtype)
            else: # Should not be reached if elements_to_read was 0 (handled above) or >0 (handled by reshape)
                return np_array_flat # Fallback for unexpected cases, will be 1D

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError(
                    "Slices into indexed_dataset must be contiguous")
            
            if start >= stop: # Empty slice
                return []

            ptr = self._index._pointers[start]
            
            # Get the logical sizes for each item in the slice
            logical_sizes_in_slice = self._index._sizes[idx] # This is a numpy array of logical token counts

            if self._index._version == 1:
                total_elements_to_read = sum(logical_sizes_in_slice)
                split_offsets_flat = list(accumulate(logical_sizes_in_slice))
            elif self._index._version == 2:
                total_elements_to_read = sum(s * self._index._token_unit_len for s in logical_sizes_in_slice)
                split_offsets_flat = list(accumulate(s * self._index._token_unit_len for s in logical_sizes_in_slice))
            else:
                raise ValueError(f"Unknown index version {self._index._version}")

            if total_elements_to_read == 0:
                # Return a list of empty 2D arrays, one for each item in the slice
                return [
                    np.empty((0, self._index._token_unit_len if self._index._version == 2 else 1), dtype=self._index.dtype)
                    for _ in logical_sizes_in_slice
                ]

            np_array_bulk_flat = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_elements_to_read, offset=ptr
            )
            
            # Split the bulk flat array into individual item flat arrays
            # np.split uses offsets relative to the start of np_array_bulk_flat
            if not split_offsets_flat or not split_offsets_flat[:-1]: # Handles single item in slice or all empty
                if total_elements_to_read > 0:
                    sents_flat = [np_array_bulk_flat]
                else: # Should be covered by total_elements_to_read == 0 above
                    sents_flat = [] 
            else:
                sents_flat = np.split(np_array_bulk_flat, split_offsets_flat[:-1])

            reshaped_sents = []
            for i, item_flat_array in enumerate(sents_flat):
                current_item_logical_size = logical_sizes_in_slice[i]
                if item_flat_array.size > 0:
                    # Expected size for this item_flat_array is current_item_logical_size * token_unit_len
                    if item_flat_array.size % self._index._token_unit_len == 0:
                        # For V1, token_unit_len is 1
                        reshaped_sents.append(item_flat_array.reshape(current_item_logical_size, self._index._token_unit_len))
                    else:
                        raise ValueError(
                            f"Item data size {item_flat_array.size} in slice is not compatible with token_unit_len {self._index._token_unit_len} for reshaping."
                        )
                elif current_item_logical_size == 0: # Logically empty item
                     reshaped_sents.append(np.empty((0, self._index._token_unit_len if self._index._version == 2 else 1), dtype=self._index.dtype))
                else: # item_flat_array.size is 0 but current_item_logical_size > 0. This indicates data inconsistency or mmap read issue.
                    # However, if total_elements_to_read was calculated correctly, and split was correct, this branch might be unlikely for valid data.
                    # For robustness, treat as empty if flat array is empty.
                    reshaped_sents.append(np.empty((0, self._index._token_unit_len if self._index._version == 2 else 1), dtype=self._index.dtype))
            
            return reshaped_sents

    def get(self, idx, offset=0, length=0): # length defaults to 0, expecting caller to provide it for meaningful read
        """Retrieves an item or a portion of an item from the dataset, with 'free addressing'.

        The \`offset\` and \`length\` parameters are in units of **logical tokens**.
        \'idx\` is used to get the starting byte pointer of the item/stream.
        The size of the item specified by \`idx\` in the .idx file does NOT cap the read length.
        Reading beyond the end of the mmaped .bin file will result in an error from np.frombuffer.

        The returned array will have shape \`[num_logical_tokens_read, token_unit_len]\`.
        """
        item_start_byte_ptr, _ = self._index[idx] # The item_total_logical_tokens from index is NOT used to cap length here.
        
        token_unit_len = self._index._token_unit_len # Number of base dtype elements per logical token
        dtype_byte_size = self._index._dtype_size    # Size of one base dtype element in bytes

        # Ensure offset and length are non-negative logical token counts.
        # User (e.g., MyDataset) is responsible for providing sensible offset and length.
        final_offset_logical_tokens = max(0, offset if offset is not None else 0)
        num_logical_tokens_to_read = max(0, length if length is not None else 0)

        # Calculate the byte offset into the .bin file to start reading from.
        # This is: item's starting byte pointer + 
        #           (logical_token_offset * elements_per_logical_token * bytes_per_element)
        start_read_byte_offset = item_start_byte_ptr + (final_offset_logical_tokens * token_unit_len * dtype_byte_size)

        # Calculate the total number of base dtype elements to read for np.frombuffer's `count` argument.
        num_base_elements_to_read = num_logical_tokens_to_read * token_unit_len

        # Read the data from the mmap.
        # np.frombuffer will raise an error if start_read_byte_offset is out of bounds
        # or if num_base_elements_to_read causes an out-of-bounds read from that offset
        # relative to the entire mmaped .bin file.
        np_array_flat = np.frombuffer(
            self._bin_buffer,
            dtype=self._index.dtype,
            count=num_base_elements_to_read,
            offset=start_read_byte_offset
        )

        # Reshape the flat array to [num_logical_tokens_read, token_unit_len]
        # token_unit_len is guaranteed to be >= 1 by Index constructor.
        # If num_logical_tokens_to_read is 0, num_base_elements_to_read is 0, np_array_flat is empty.
        # Reshaping an empty array to (0, N) is valid.
        return np_array_flat.reshape(num_logical_tokens_to_read, token_unit_len)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
