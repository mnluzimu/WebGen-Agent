import json
import random
from collections import defaultdict

random.seed(517)

def shuffle_and_classify(input_file, output_file, remain_file, summary_file, sample_limit=70):
    # Read records from the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    # Shuffle records
    random.shuffle(records)

    # Classify records by 'application_type'
    classified_records = defaultdict(list)
    for record in records:
        app_type = record.get('application_type', 'unknown')
        classified_records[app_type].append(record)

    # Extract first 'sample_limit' samples per category and save to output file
    output_records = []
    remain_records = []
    summary = {}

    for app_type, recs in classified_records.items():
        summary[app_type] = len(recs)
        output_records.extend(recs[:sample_limit])
        remain_records.extend(recs[sample_limit:])

    output_records = sorted(output_records, key=lambda x: x["id"])

    # Save output records
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Save remaining records
    with open(remain_file, 'w', encoding='utf-8') as f:
        for record in remain_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Save summary to JSON file
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    input_file = '/mnt/cache/luzimu/code_agent/data/WebGen-Bench/train.jsonl'
    output_file = '/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample/WebGen-Instruct_1.jsonl'
    remain_file = '/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample/WebGen-Instruct_remain_1.jsonl'
    summary_file = '/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample/summary_1.jsonl'

    shuffle_and_classify(input_file, output_file, remain_file, summary_file, sample_limit=20)


# Example usage
if __name__ == "__main__":
    main()
