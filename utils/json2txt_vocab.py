import json

input_path =  "assets/vocab/prot/vocab_4096/vocab.json"
output_path = "assets/vocab/prot/vocab_4096/vocab.txt"

# 读取 vocab.json
with open(input_path, "r", encoding="utf-8") as f:
    vocab_dict = json.load(f)

# 构造 id -> token 映射
max_id = max(vocab_dict.values())
vocab = [None] * (max_id + 1)
for token, idx in vocab_dict.items():
    vocab[idx] = token

# 写入 vocab.txt
with open(output_path, "w", encoding="utf-8") as f:
    for idx, token in enumerate(vocab):
        if token is None:
            continue  # 跳过空洞 id（理论上不应该有）

        # 获取 UTF-8 字节长度
        token_bytes = token.encode("utf-8")
        byte_len = len(token_bytes)

        # 转换为单引号包裹的 Python 样式字符串，例如 'a'、'\n'、'\xE4'
        token_repr = repr(token)

        # 写入一行
        f.write(f"{idx} {token_repr} {byte_len}\n")

print(f"✅ vocab.json 转换完成 -> 写入 {output_path}")