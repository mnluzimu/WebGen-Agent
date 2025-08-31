import json
import os 
import random
from collections import defaultdict


def save_jsonl(datas, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def sample_from_top_matches(input_file, output_file, sample_limit=20):
    with open(input_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    classified_records = defaultdict(list)
    classified_instructions = defaultdict(list)

    for data in datas:
        matches = data.get("matches", [])
        for match in matches:
            app_type = match.get('info', {}).get('application_type', 'unknown')
            if match.get('train_instruction') not in classified_instructions[app_type]:
                classified_instructions[app_type].append(match['train_instruction'])
                classified_records[app_type].append(match)
    
    outputs = []
    counts = {app_type: len(recs) for app_type, recs in classified_records.items()}
    counts["total"] = sum(counts.values())
    for app_type, recs in classified_records.items():
        random.shuffle(recs)
        outputs.extend([e["info"] for e in recs[:sample_limit]])

    outputs = sorted(outputs, key=lambda x: x["id"])

    with open(os.path.join(os.path.dirname(output_file), 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(counts, f, ensure_ascii=False, indent=4)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_file = '/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample_rl/top_matches.json'
    output_file = '/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample_rl/train_select_2.jsonl'
    
    sample_from_top_matches(input_file, output_file, sample_limit=30)
    print(f"Sampled data saved to {output_file}")

    