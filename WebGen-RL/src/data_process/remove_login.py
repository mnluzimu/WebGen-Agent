import json
import os
from tqdm import tqdm


def save_jsonl(datas, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for data in tqdm(datas):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def remove_login_info(input_file, output_file):
    """
    Remove login information from the input JSONL file and save to the output file.
    """
    with open(input_file, 'r', encoding="utf-8") as f:
        datas = json.load(f)

    for data in datas:
        data["instruction"] = data["instruction"].replace("log in, ", "")\
        .replace("log in and ", "").replace("log in to the system, ", "")\
        .replace("log in to the system and ", "")\
        .replace("log in to the platform, ", "")\
        .replace("log in to the platform and ", "")\
        .replace("register and log in to the website, ", "")\
        .replace("register and log in to their accounts, ", "")\
        .replace("log in to the website, ", "")\
        .replace("register and log in to the mini-program, ", "")\
        .replace("log in to the mini-program, ", "")\
        .replace("register and log in to the application, ", "")\
        .replace("log in to the application, ", "")\
        .replace("log in to the mini-program and ", "")\
        .replace("register and log in to the app, ", "")\
        .replace(", user registration, and login", "")\
        .replace("user login, ", "")\
        .replace("user login, registration, and ", "")\
        .replace("user login, registration, ", "")\
        .replace("user registration, login, and ", "")\
        .replace("user registration, login, ", "")\
        .replace("registration, login, ", "")\
        .replace("login, registration, ", "")\
        .replace("registration and login, ", "")\
        .replace("login, ", "")

    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)
    save_jsonl(datas, output_file.replace(".json", ".jsonl"))


if __name__ == "__main__":
    input_file = "/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample_rl/train_select_2.json"
    output_file = "/mnt/cache/luzimu/code_agent/data/WebGen-Bench/sample_rl/train_select_3.json"
    
    remove_login_info(input_file, output_file)
    print(f"Login information removed and saved to {output_file}")