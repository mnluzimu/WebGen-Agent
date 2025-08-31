import os

from datasets import Dataset
import argparse
from tqdm import tqdm
import json 

from system import system_prompt


def load_jsonl(in_file: str):
    """Safely read a JSONL file into a list (empty list if missing)."""
    if not os.path.isfile(in_file):
        return []

    data = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()    
    parser.add_argument('--input',type=str, default='/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample_rl/train_select_3.jsonl')
    parser.add_argument('--output', type=str, default='/mnt/cache/luzimu/code_agent/data/WebGen-Instruct_2_501')
    args=parser.parse_args()

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(system_prompt)

    # read through the json file 
    output_dataset=[]
    split = 'train'
    i = 0
    input_dataset = load_jsonl(args.input)
        
    for data_entry in tqdm(input_dataset):
        cur_data = {
            "data_source": "webgen",
            "prompt": [{
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": data_entry["instruction"]
            }],
            "ability": 'webgen',
            "reward_model": {
                "style": "rule",
                "ground_truth": "",
            },
            "extra_info": {
                'split': 'dummy',
                'index': i
            }
        }
        output_dataset.append(cur_data)
        i += 1 
        
    print(len(output_dataset))
    output_dataset = Dataset.from_list(output_dataset)
    output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
    
    split = 'validation'
    i = 0
    output_dataset = [] 
            
    for data_entry in tqdm(input_dataset):
        cur_data = {
            "data_source": "webgen",
            "prompt": [{
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": data_entry["instruction"]
            }],
            "ability": 'webgen',
            "reward_model": {
                "style": "rule",
                "ground_truth": "",
            },
            "extra_info": {
                'split': 'dummy',
                'index': i
            }
        }
        output_dataset.append(cur_data)
        i += 1
        break 
        
    print(len(output_dataset))
    output_dataset = Dataset.from_list(output_dataset)
    output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
