"""
OpenAI-compatible API inference (route: api).
Single script for OpenAI, Ollama, vLLM: client = OpenAI(base_url=..., api_key=...); chat.completions.create(...).
Use --backend openai | ollama | vllm to set base_url and api_key.
"""
import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from openai import OpenAI
from utils.dataset import OMGDataset, TMGDataset

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="openai", choices=("openai", "ollama", "vllm", "api"),
                    help="API backend: openai (default), ollama (localhost:11434), vllm (localhost:port); api=openai")
parser.add_argument("--name", type=str, default="GPT-4o")
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--port", type=int, default=8000, help="vLLM server port when backend=vllm")
parser.add_argument("--base_url", type=str, default=None, help="Override API base URL (e.g. custom proxy)")
parser.add_argument("--api_key", type=str, default=None, help="Override API key (default: env OPENAI_API_KEY or backend default)")
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--use_hf", action="store_true", default=True)
parser.add_argument("--no_use_hf", action="store_false", dest="use_hf")
parser.add_argument("--output_dir", type=str, default="./predictions/")
parser.add_argument("--temperature", type=float, default=0.75)
parser.add_argument("--top_p", type=float, default=0.85)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timeout", type=float, default=60.0)
parser.add_argument("--json_check", action="store_true", default=False)
parser.add_argument("--smiles_check", action="store_true", default=False)
parser.add_argument("--log", action="store_true", default=False)
parser.add_argument("--save_json", action="store_true")
parser.add_argument("--eval_after", action="store_true")

args = parser.parse_args()

# Resolve backend alias: api -> openai
if args.backend == "api":
    args.backend = "openai"
# Resolve base_url and api_key by backend
if args.base_url is None:
    if args.backend == "openai":
        args.base_url = os.environ.get("OPENAI_API_BASE") or None
    elif args.backend == "ollama":
        args.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
    else:
        args.base_url = "http://localhost:{}/v1".format(args.port)
if args.api_key is None:
    if args.backend == "openai":
        args.api_key = os.environ.get("OPENAI_API_KEY", "")
    elif args.backend == "ollama":
        args.api_key = "ollama"
    else:
        args.api_key = "EMPTY"

client = OpenAI(base_url=args.base_url, api_key=args.api_key)

# output dir and file
args.output_dir = args.output_dir.rstrip("/") + "/" + args.name + "/" + args.benchmark + "/" + args.task + "/"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
out_suffix = args.subtask + ".csv"

if os.path.exists(args.output_dir + out_suffix):
    temp = pd.read_csv(args.output_dir + out_suffix)
    start_pos = len(temp)
else:
    with open(args.output_dir + out_suffix, "w+") as f:
        f.write("outputs\n")
    start_pos = 0

print("========Parameters========")
for attr, value in args.__dict__.items():
    print("{}={}".format(attr.upper(), value))
print("========Inference Init========")
print("Inference starts from: ", start_pos)

if args.benchmark == "open_generation":
    inference_dataset = OMGDataset(args.task, args.subtask, args.json_check, use_hf=args.use_hf, data_dir=args.data_dir)
elif args.benchmark == "targeted_generation":
    inference_dataset = TMGDataset(args.task, args.subtask, args.json_check, use_hf=args.use_hf, data_dir=args.data_dir)
print("========Sanity Check========")
print(inference_dataset[0])
print("Total length of the dataset:", len(inference_dataset))
print("==============================")

error_records = []

# Common completion kwargs (OpenAI-compatible)
completion_kwargs = {
    "model": args.model,
    "max_tokens": args.max_new_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "n": args.num_return_sequences,
    "timeout": args.timeout,
}
if args.backend == "vllm":
    completion_kwargs["stop"] = ["</s>", "<|end_of_text|>", "<|eot_id|>"]
    completion_kwargs["seed"] = args.seed

with tqdm(total=len(inference_dataset) - start_pos) as pbar:
    for idx in range(start_pos, len(inference_dataset)):
        cur_seed = args.seed
        error_allowance = 0
        while True:
            try:
                messages = inference_dataset[idx]
                completion = client.chat.completions.create(messages=messages, **completion_kwargs)
                s = completion.choices[0].message.content
            except Exception as e:
                print("Error:", e)
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = "None"
                    error_records.append(idx)
                    break
                continue

            s = s.replace('""', '"').strip() if s else ""
            print("Raw:", s)

            if s is None or (isinstance(s, str) and not s.strip()):
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = ""
                    error_records.append(idx)
                    break
                continue

            if args.log:
                with open(args.output_dir + out_suffix.replace(".csv", ".log"), "a+") as f:
                    f.write(s.replace("\n", " ").strip() + "\n")

            if args.json_check:
                match = re.search(r'\{.*?\}', s, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        obj = json.loads(json_str)
                        s = obj.get("molecule", s)
                        if args.smiles_check:
                            try:
                                mol = Chem.MolFromSmiles(s)
                                if mol is None:
                                    cur_seed += 1
                                    error_allowance += 1
                                    if error_allowance > 10:
                                        error_records.append(idx)
                                        break
                                    continue
                            except Exception:
                                cur_seed += 1
                                error_allowance += 1
                                if error_allowance > 10:
                                    error_records.append(idx)
                                    break
                                continue
                        break
                    except Exception:
                        cur_seed += 1
                        error_allowance += 1
                        if error_allowance > 10:
                            error_records.append(idx)
                            break
                        continue
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    error_records.append(idx)
                    break
                continue
            break

        print("Checked:", s)
        if not isinstance(s, str):
            s = str(s)
        s = s.replace("\n", " ").strip()

        df = pd.DataFrame([s.strip()], columns=["outputs"])
        df.to_csv(args.output_dir + out_suffix, mode="a", header=False, index=True)
        pbar.update(1)

print("========Inference Done========")
print("Error Records: ", error_records)

if args.save_json:
    pred_df = pd.read_csv(args.output_dir + out_suffix)
    outputs_list = pred_df["outputs"].astype(str).tolist()
    records = []
    for i in range(len(outputs_list)):
        msg = inference_dataset[i]
        inst = msg[1].get("content", msg[1]) if isinstance(msg[1], dict) else msg[1]
        records.append({"idx": i, "instruction": inst, "output": outputs_list[i]})
    out = {
        "model_name": args.name,
        "task": args.task,
        "subtask": args.subtask,
        "benchmark": args.benchmark,
        "num_samples": len(records),
        "outputs": records,
    }
    json_path = args.output_dir + out_suffix.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved JSON:", json_path)

if args.eval_after:
    import subprocess
    import sys
    base_dir = os.path.normpath(os.path.join(args.output_dir, "..", "..", ".."))
    subprocess.run([
        sys.executable, "evaluate.py",
        "--name", args.name,
        "--benchmark", args.benchmark,
        "--task", args.task,
        "--subtask", args.subtask,
        "--output_dir", base_dir + os.sep,
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
