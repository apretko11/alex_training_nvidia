import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
import datasets
from datasets import load_dataset


ds_id = "NadineMostafa/AnghaBench-armv8-O2-native-clang-20percent-shard3"
MAX_CONTEXT_LENGTH = 16_000
CONTEXT_LENGTH_OFFSET = 1000
SYSTEM_PROMPT = """You are a helpful coding assistant specialized in converting from x86 to ARM assembly."""
USER_PROMPT = """Convert the following x86 assembly code to ARM assembly:\n```x86asm\n{asm}```"""
OUTPUT_PROMPT = """```armasm\n{asm}```"""
dataset_name = "AnghaBench-armv8-O2-native-clang-20percent_3"

# Load dataset with pandas and convert to Dataset
print(f"Loading dataset from: {ds_id}")
#df = pd.read_parquet(ds_id)
#ds = datasets.Dataset.from_pandas(df)
ds = load_dataset(ds_id, split="train")


def process_example(example):
    x86 = example["x86_content"]
    arm = example["armv8_content"]
    if len(x86 + arm) + CONTEXT_LENGTH_OFFSET > MAX_CONTEXT_LENGTH:
        return None

    return {
        "conversations": [
            {
                "role": "user",
                "content": USER_PROMPT.format(asm=x86)
            },
            {
                "role": "assistant",
                "content": OUTPUT_PROMPT.format(asm=arm)
            }
        ],
        "system": SYSTEM_PROMPT,
    }

# Process the data
print("Processing examples...")
with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_example, ds), total=len(ds), desc="Processing examples"))

data = [result for result in results if result is not None]

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

total_examples = len(data)
print(f"Total examples: {total_examples}")

# Create single JSON file
filename = f"data/{dataset_name}.json"

with open(filename, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Created {filename} with {total_examples} examples")

# Handle dataset_info.json safely
dataset_info_path = f"data/dataset_info.json"
if os.path.exists(dataset_info_path):
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
else:
    dataset_info = {}

# Add entry for the single file
dataset_info[dataset_name] = {
    "file_name": f"{dataset_name}.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "system": "system",
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant"
    }
}

with open(dataset_info_path, "w") as f:
    json.dump(dataset_info, f, indent=4, ensure_ascii=False)

print("Dataset processing complete!")
print(f"Created single JSON file and updated {dataset_info_path}")
