"""
Local Hugging Face inference (route: hf).
- model_type=causal (default): CausalLM with chat format, optional LoRA. Single/multi-GPU via device_map.
- model_type=t5: T5 or encoder-decoder (MolT5, BioT5). Use --selfies for BioT5 (SELFIES in/out).
- model_type=decoder-only: adapter-based decoder (e.g. Galactica from checkpoint). Plain text in, no chat template.
Other route: query_api.py (--backend openai|ollama|vllm for OpenAI-compatible APIs).
"""
import os
import re
import json
import random
import rdkit
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from utils.dataset import OMGDataset, TMGDataset, OMGInsTDataset
import transformers
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, GenerationConfig
from accelerate import dispatch_model, infer_auto_device_map
from accelerate import Accelerator
from accelerate.utils import gather_object

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="causal", choices=("causal", "t5", "decoder-only"),
                    help="causal=chat CausalLM; t5=MolT5/BioT5 (use --selfies for BioT5); decoder-only=adapter CausalLM")
parser.add_argument("--model", type=str, default="quantized_models/llama3-70b/", help="CausalLM path (causal only)")
parser.add_argument("--name", type=str, default="llama3-70B")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--load_lora", type=bool, default=False)
parser.add_argument("--lora_model_path", type=str, default="")
# T5 / decoder-only: load from adapter (and optional base for LoRA)
parser.add_argument("--base_model", type=str, default="", help="Tokenizer / base for LoRA (t5/decoder-only)")
parser.add_argument("--adapter_path", type=str, default="", help="T5 or full fine-tuned checkpoint (t5/decoder-only)")
parser.add_argument("--enable_lora", action="store_true", help="Decoder-only: load base + LoRA from adapter_path")
parser.add_argument("--int4", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--selfies", action="store_true", help="Use SELFIES in instructions (BioT5); OMGDataset use_selfies=True")
parser.add_argument("--partition", type=int, default=1)
parser.add_argument("--cur", type=int, default=1)
# dataset
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")
parser.add_argument("--data_dir", type=str, default=None, help="Local benchmark data dir (e.g. full/ or benchmarks/)")
parser.add_argument("--use_hf", action="store_true", default=True, help="Load benchmark from Hugging Face first")
parser.add_argument("--no_use_hf", action="store_false", dest="use_hf", help="Load only from local data_dir")
parser.add_argument("--benchmark_scale", type=str, default="full", choices=("full", "mini"),
                    help="S²-Bench: full or mini (phenixace/S2-TOMG-Bench or S2-TOMG-Bench-mini)")

parser.add_argument("--output_dir", type=str, default="./predictions/")

parser.add_argument("--temperature", type=float, default=0.75)
parser.add_argument("--top_p", type=float, default=0.85)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)

parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--json_check", action="store_true", default=False)
parser.add_argument("--smiles_check", action="store_true", default=False)

parser.add_argument("--log", action="store_true", default=False)
parser.add_argument("--save_json", action="store_true", help="Also save results to one JSON file per (task, subtask)")
parser.add_argument("--eval_after", action="store_true", help="Run evaluate.py after inference and print metrics")

args = parser.parse_args()

if "mistral" in args.model:
    args.mistral = True
else:
    args.mistral = False

# output file: subtask.csv or subtask_cur.csv when partition/cur used (t5-style)
out_suffix = args.subtask + "_" + str(args.cur) + ".csv" if (args.model_type != "causal" and (args.partition != 1 or args.cur != 1)) else args.subtask + ".csv"

# print parameters
print("========Parameters========")
for attr, value in args.__dict__.items():
    print("{}={}".format(attr.upper(), value))

# output dir
args.output_dir = args.output_dir.rstrip("/") + "/" + args.name + "/" + args.benchmark + "/" + args.task + "/"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if os.path.exists(args.output_dir + out_suffix):
    temp = pd.read_csv(args.output_dir + out_suffix)
    start_pos = len(temp)
else:
    with open(args.output_dir + out_suffix, "w+") as f:
        f.write("outputs\n")
    start_pos = 0

print("========Inference Init========")
print("Inference starts from: ", start_pos)

# -------- T5 / decoder-only: text-in path (no chat template) --------
if args.model_type in ("t5", "decoder-only"):
    if args.benchmark == "open_generation":
        if args.model_type == "t5":
            inference_dataset = OMGDataset(args.task, args.subtask, json_check=False, use_selfies=args.selfies, use_hf=args.use_hf, data_dir=args.data_dir, hf_benchmark_repo=args.benchmark_scale)
        else:
            inference_dataset = OMGInsTDataset(args.task, args.subtask, use_hf=args.use_hf, data_dir=args.data_dir, hf_benchmark_repo=args.benchmark_scale)
    elif args.benchmark == "targeted_generation":
        inference_dataset = TMGDataset(args.task, args.subtask, json_check=False, use_hf=args.use_hf, data_dir=args.data_dir, hf_benchmark_repo=args.benchmark_scale)
    else:
        raise ValueError("Invalid benchmark: {}".format(args.benchmark))

    tokenizer_path = args.base_model or args.adapter_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    device_map = "auto"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    transformers.set_seed(args.seed)
    random.seed(args.seed)

    if args.model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(args.adapter_path, device_map=device_map)
    else:
        if args.enable_lora:
            from peft import PeftModel
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model or args.adapter_path,
                load_in_8bit=args.int8, load_in_4bit=args.int4,
                torch_dtype=torch.float16 if args.fp16 else torch.float32,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=torch.float16 if args.fp16 else torch.float32)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.adapter_path,
                load_in_4bit=args.int4, load_in_8bit=args.int8,
                torch_dtype=torch.float16 if args.fp16 else torch.float32,
                device_map=device_map,
            )
    model = model.half() if not args.fp16 else model
    model.eval()

    gen_config = GenerationConfig(
        do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        num_beams=args.num_beams, pad_token_id=tokenizer.pad_token_id or 0,
    )

    start_i = int(len(inference_dataset) * (args.cur - 1) / args.partition) + start_pos
    end_i = int(len(inference_dataset) * args.cur / args.partition)
    error_records = []

    for idx in range(start_i, end_i):
        if args.model_type == "t5":
            text_input = inference_dataset.instructions[idx]
        else:
            # decoder-only: OMGInsTDataset has .data; TMGDataset has .instructions
            text_input = inference_dataset.data[idx][0] if hasattr(inference_dataset, "data") else inference_dataset.instructions[idx]
        model_input = tokenizer(text_input, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            gen = model.generate(inputs=model_input, generation_config=gen_config, return_dict_in_generate=True,
                                 max_new_tokens=args.max_new_tokens, num_return_sequences=args.num_return_sequences)
        s = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
        if args.model_type == "decoder-only" and "## Molecule:" in s:
            s = s.split("## Molecule:")[-1].strip()
        s = s.replace("\n", " ").strip()
        df = pd.DataFrame([s], columns=["outputs"])
        df.to_csv(args.output_dir + out_suffix, mode="a", header=False, index=False)

    print("========Inference Done========")
    print("Error Records: ", error_records)

    if args.save_json:
        pred_df = pd.read_csv(args.output_dir + out_suffix)
        outputs_list = pred_df["outputs"].astype(str).tolist()
        records = []
        for idx in range(len(outputs_list)):
            if args.model_type == "t5":
                inst = inference_dataset.instructions[idx]
            else:
                inst = inference_dataset._hf_ds["Instruction"][idx] if hasattr(inference_dataset, "_hf_ds") else inference_dataset.data[idx][0]
            records.append({"idx": idx, "instruction": inst, "output": outputs_list[idx]})
        out = {"model_name": args.name, "task": args.task, "subtask": args.subtask, "benchmark": args.benchmark, "num_samples": len(records), "outputs": records}
        json_path = args.output_dir + out_suffix.replace(".csv", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Saved JSON:", json_path)

    if args.eval_after:
        import subprocess, sys
        base_dir = os.path.normpath(os.path.join(args.output_dir, "..", "..", ".."))
        subprocess.run([sys.executable, "evaluate.py", "--name", args.name, "--benchmark", args.benchmark,
                        "--task", args.task, "--subtask", args.subtask, "--output_dir", base_dir + os.sep,
                        "--predictions", args.output_dir + out_suffix], cwd=os.path.dirname(os.path.abspath(__file__)))

    raise SystemExit(0)

# -------- CausalLM chat path --------
if args.benchmark == "open_generation":
    inference_dataset = OMGDataset(args.task, args.subtask, args.json_check, use_hf=args.use_hf, data_dir=args.data_dir, hf_benchmark_repo=args.benchmark_scale)
elif args.benchmark == "targeted_generation":
    inference_dataset = TMGDataset(args.task, args.subtask, args.json_check, use_hf=args.use_hf, data_dir=args.data_dir, hf_benchmark_repo=args.benchmark_scale)
print("========Sanity Check========")
print(inference_dataset[0])
print("Total length of the dataset:", len(inference_dataset))
print("==============================")

error_records = []

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size > 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

if args.load_lora == True:
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager", device_map=device_map, trust_remote_code=True).eval()
    print(f"Loading LoRA weights from {args.lora_model_path}")
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    print(f"Merging weights")
    model = model.merge_and_unload()
    print('Convert to BF16...')
    model = model.to(torch.bfloat16)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

if ddp:
    accelerator = Accelerator()
    model = accelerator.prepare(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=args.model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    temperature=args.temperature,
    trust_remote_code=True,
    top_p=args.top_p,
)

def is_main_process():
    return not ddp or (ddp and accelerator.is_main_process)

with tqdm(total=len(inference_dataset)-start_pos, disable=not is_main_process()) as pbar:
    for idx in range(start_pos, len(inference_dataset)):
        if ddp:
            accelerator.wait_for_everyone()
        cur_seed = args.seed
        error_allowance = 0
        while True:
            try:
                prompt = inference_dataset[idx]
                inputs = tokenizer.apply_chat_template(prompt,
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True
                                            )
                inputs = inputs.to(model.device)
                gen_kwargs = {"max_length": args.max_new_tokens, "do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    s = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if ddp:
                    s = gather_object([s])[0]
            except:
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = "None"
                    error_records.append(idx)
                    break
                else:
                    continue
            
            s = s.replace('""', '"').strip()
            print("Raw:", s)

            if s == None:
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = ""
                    error_records.append(idx)
                    break
                else:
                    continue

            if args.log:
                with open(args.output_dir + out_suffix.replace(".csv", ".log"), "a+") as f:
                    f.write(s.replace('\n', ' ').strip() + "\n")

            if args.json_check:
                match = re.search(r'\{.*?\}', s, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        json_obj = json.loads(json_str)
                        s = json_obj["molecule"]
                        if args.smiles_check:
                            try:
                                mol = Chem.MolFromSmiles(s)
                                if mol is None:
                                    cur_seed += 1
                                    error_allowance += 1
                                    if error_allowance > 10:
                                        error_records.append(idx)
                                        break
                                    else:
                                        continue
                            except:
                                cur_seed += 1
                                error_allowance += 1
                                if error_allowance > 10:
                                    error_records.append(idx)
                                    break
                                else:
                                    continue
                        break
                    except:
                        cur_seed += 1
                        error_allowance += 1
                        if error_allowance > 10:
                            error_records.append(idx)
                            break
                        else:
                            continue

                else:
                    cur_seed += 1
                    error_allowance += 1
                    if error_allowance > 10:
                        error_records.append(idx)
                        break
                    else:
                        continue
            else:
                break
        print("Checked:", s)
        
        if not isinstance(s, str):
            s = str(s)

        s = s.replace('\n', ' ').strip()

        if is_main_process():
            df = pd.DataFrame([s.strip()], columns=["outputs"])
            df.to_csv(args.output_dir + out_suffix, mode='a', header=False, index=True)
        pbar.update(1)


if is_main_process():
    print("========Inference Done========")
    print("Error Records: ", error_records)

if is_main_process() and args.save_json:
    pred_df = pd.read_csv(args.output_dir + out_suffix)
    outputs_list = pred_df["outputs"].astype(str).tolist()
    records = []
    for idx in range(len(outputs_list)):
        msg = inference_dataset[idx]
        inst = msg[1].get("content", msg[1]) if isinstance(msg[1], dict) else msg[1]
        records.append({"idx": idx, "instruction": inst, "output": outputs_list[idx]})
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

if is_main_process() and args.eval_after:
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
