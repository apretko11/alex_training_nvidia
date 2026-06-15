import json
import os
from tqdm import tqdm
from datasets import load_dataset

# ==== Config ====
ds_id = "adpretko/x86-to-llvm-o0"
MAX_CONTEXT_LENGTH = 16_000
CONTEXT_LENGTH_OFFSET = 1000
SYSTEM_PROMPT = "You are a helpful coding assistant specialized in converting from x86 assembly to LLVM IR."
USER_PROMPT   = "Convert the following x86 assembly code to LLVM IR:\n```x86asm\n{asm}```"
OUTPUT_PROMPT = "```llvm\n{asm}```"

dataset_name = "x86-to-llvm-o0"
OUT_DIR = os.environ.get("OUT_DIR", "data")
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "50000"))  # change via env if needed

# ==== Load dataset ====
print(f"Loading dataset from: {ds_id}")
ds = load_dataset(ds_id, split="train", streaming=True)

# sanity: show columns once
print("Dataset columns:", ds.column_names)

def process_example(example):
    # If your dataset uses different keys, adjust here.
    x86  = example["x86"]
    llvm = example["llvm"]
    if len(x86 + llvm) + CONTEXT_LENGTH_OFFSET > MAX_CONTEXT_LENGTH:
        return None
    return {
        "conversations": [
            {"role": "user", "content": USER_PROMPT.format(asm=x86)},
            {"role": "assistant", "content": OUTPUT_PROMPT.format(asm=llvm)},
        ],
        "system": SYSTEM_PROMPT,
    }

# ==== Ensure output dir ====
os.makedirs(OUT_DIR, exist_ok=True)

# ==== Stream -> shards (array-per-file JSON, compact) ====
print("Processing & sharding examples...")
buffer = []
total_written = 0
shard_idx = 0
shard_files = []

def write_shard(items, idx):
    if not items:
        return None
    fname = f"{dataset_name}.{idx:02d}.json"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w") as f:
        json.dump(items, f, ensure_ascii=False)  # no indent to save space
    return fname, len(items)

for ex in tqdm(ds, desc="Processing examples"):
    item = process_example(ex)
    if item is None:
        continue
    buffer.append(item)
    if len(buffer) >= SHARD_SIZE:
        fname, n = write_shard(buffer, shard_idx)
        shard_files.append(fname)
        total_written += n
        buffer.clear()
        shard_idx += 1

# tail shard
if buffer:
    fname, n = write_shard(buffer, shard_idx)
    shard_files.append(fname)
    total_written += n

print(f"Total examples written across shards: {total_written}")
print("Shard files:")
for s in shard_files:
    print("  ", os.path.join(OUT_DIR, s))

# ==== Update dataset_info.json ====
dataset_info_path = os.path.join(OUT_DIR, "dataset_info.json")
if os.path.exists(dataset_info_path):
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
else:
    dataset_info = {}

dataset_info[dataset_name] = {
    "file_name": shard_files,   # list of shard filenames
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}

with open(dataset_info_path, "w") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("Done. Updated", dataset_info_path)

