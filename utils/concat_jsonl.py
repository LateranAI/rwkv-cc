import subprocess
import time
import os
from tqdm import tqdm


def count_lines(file_path: str) -> int:
    """快速统计文件行数（用于小文件）"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


def append_jsonl_with_progress(input_path: str, output_file):
    """将小文件追加到输出文件，并带进度条"""
    print(f"[INFO] Counting lines in small file: {input_path}")
    total_lines = count_lines(input_path)
    print(f"[INFO] Total lines in small file: {total_lines}")

    print(f"[INFO] Appending {input_path} → output (with progress)")
    with open(input_path, 'r', encoding='utf-8') as in_file, \
            tqdm(total=total_lines, desc="Appending", unit="lines") as pbar:
        for line in in_file:
            output_file.write(line)
            pbar.update(1)
    print(f"[INFO] Finished appending {input_path}")


def fast_copy_with_rsync(source_path: str, target_path: str):
    """使用 rsync 执行高性能文件复制"""
    print(f"[INFO] Copying large file using rsync: {source_path} → {target_path}")
    start = time.time()
    result = subprocess.run(["rsync", "-a", "--info=progress2", source_path, target_path])
    if result.returncode != 0:
        raise RuntimeError(f"[ERROR] rsync failed with return code {result.returncode}")
    elapsed = time.time() - start
    print(f"[INFO] rsync copy complete in {elapsed:.2f} seconds.")


def concat_large_and_small_jsonl(large_path: str, small_path: str, output_path: str):
    start_time = time.time()

    # 拷贝大文件
    fast_copy_with_rsync(large_path, output_path)
    print(f"[INFO] Copied file size: {os.path.getsize(output_path) / (1024 ** 3):.2f} GB")

    # 追加小文件
    with open(output_path, 'a', encoding='utf-8') as out_file:
        append_jsonl_with_progress(small_path, out_file)

    print(f"[SUCCESS] Merging complete. Output saved to: {output_path}")
    print(f"[TIME] Total time elapsed: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    large_file = r"/public/home/ssjxzkz/Datasets/uniprot/uniprot_trembl.jsonl"
    small_file = r"/public/home/ssjxzkz/Datasets/uniprot/uniprot_sprot.jsonl"
    output_file = r"/public/home/ssjxzkz/Datasets/uniprot/uniprot_merged.jsonl"

    concat_large_and_small_jsonl(large_file, small_file, output_file)