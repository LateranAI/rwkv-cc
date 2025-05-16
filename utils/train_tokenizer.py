import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm

input_files = [
    r"/public/home/ssjxzkz/Datasets/uniprot/uniprot_sprot.jsonl",
    r"/public/home/ssjxzkz/Datasets/uniprot/uniprot_trembl.jsonl",
]
output_dir = "assets"
# vocab_sizes = [65536, 32768, 16384, 8192, 4096]
vocab_sizes = [4096]
sample_ratio = 0.02  # 采样比例：比如 0.1 表示只用 10% 的数据
block_size = 1000  # 每次读取的行数块大小


def read_jsonl(file_paths: list[str], sample_ratio: float, block_size: int):
    for file_path in file_paths:
        print(f"\n开始处理文件: {os.path.basename(file_path)}")
        file_size = os.path.getsize(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"处理进度 ({os.path.basename(file_path)})",
                    dynamic_ncols=True,
            ) as pbar:
                block = []
                for line in f:
                    pbar.update(len(line.encode("utf-8")))
                    line = line.strip()
                    if line:
                        block.append(line)

                    if len(block) >= block_size:
                        yield from process_block(block, sample_ratio)
                        block.clear()

                # 处理最后一块不足 block_size 的数据
                if block:
                    yield from process_block(block, sample_ratio)


def process_block(block: list[str], sample_ratio: float):
    # 从每个 block 中随机选择 sample_ratio 百分比的样本
    sampled_block = random.sample(block, int(len(block) * sample_ratio))
    for line in sampled_block:
        try:
            data = json.loads(line)
            if "text" in data:
                yield data["text"]
            else:
                print(f"\nwarning: Missing 'text' Key")
        except json.JSONDecodeError:
            print(f"\nwarning: JSON Parse Error")


def train_tokenizer(vocab_size: int, sample_ratio: float):
    data_iterator = read_jsonl(input_files, sample_ratio, block_size)

    tokenizer = ByteLevelBPETokenizer()
    vocab_path = "assets/vocab_65536/vocab.json"
    merges_path = "assets/vocab_65536/vocab.json"
    tokenizer = tokenizer.from_file(vocab_path, merges_path)
    tokenizer.train_from_iterator(
        data_iterator,
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=[],
    )

    model_dir = os.path.join(output_dir, f"vocab_{vocab_size}")
    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_model(model_dir)

    return f"Vocab size: {vocab_size} training has been completed，the result saved to {model_dir}"


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=len(vocab_sizes)) as executor:
        future_to_vocab_size = {executor.submit(train_tokenizer, size, sample_ratio): size for size in vocab_sizes}

        for future in as_completed(future_to_vocab_size):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Vocab size: {future_to_vocab_size[future]} training error: {e}")