"""
Evaluation script. Benchmark data is loaded via Hugging Face datasets.
Optional --correct: post-process raw prediction text (extract SMILES from JSON/=>/->, optional SELFIES decode).
"""
import argparse
import re
import json
import pandas as pd
from tqdm import tqdm

from utils.dataset import get_benchmark_hf_dataset
from utils.evaluation import mol_prop, calculate_novelty, calculate_similarity


def correct_text(text):
    """Extract valid SMILES from raw model output (JSON, =>, ->, etc.)."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            json_str = match.group().replace('""', '"')
            try:
                obj = json.loads(json_str)
                s = obj.get("molecule", json_str)
            except Exception:
                s = json_str.split(":")[1].strip().strip('}').strip().strip('"').strip()
            for sep in ["=>", "->"]:
                if sep in s:
                    s = s.split(sep)[-1].strip()
            return s.strip()
        s = text.replace('\n', ' ').strip()
        for sep in ["=>", "->"]:
            if sep in s:
                s = s.split(sep)[-1].strip()
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            s = s[1:-1]
        return s
    except Exception:
        return "None"


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="galactica-125M-xlarge")

# dataset settings
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")
parser.add_argument("--data_dir", type=str, default=None, help="Local benchmark data dir (e.g. full/ or benchmarks/)")
parser.add_argument("--use_hf", action="store_true", default=True, help="Load benchmark from Hugging Face Hub first")
parser.add_argument("--no_use_hf", action="store_false", dest="use_hf", help="Load only from local data_dir")
parser.add_argument("--benchmark_scale", type=str, default="full", choices=("full", "mini"), help="S²-Bench scale: full or mini")

parser.add_argument("--output_dir", type=str, default="./reb2/")
parser.add_argument("--predictions", type=str, default=None, help="Path to predictions CSV or JSON (default: output_dir/name/benchmark/task/subtask.csv)")
parser.add_argument("--correct", action="store_true", help="Post-process raw predictions (extract SMILES from JSON/=>/->; decode SELFIES if name contains 'biot5')")
parser.add_argument("--calc_novelty", action="store_true", default=False)

args = parser.parse_args()

target_file = args.predictions or (args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/" + args.subtask + ".csv")

# Load benchmark ground truth via datasets
data = get_benchmark_hf_dataset(
    args.task, args.subtask, benchmark=args.benchmark,
    use_hf=args.use_hf, data_dir=args.data_dir,
    hf_benchmark_repo=args.benchmark_scale,
)
data_len = len(data)

if target_file.lower().endswith(".json"):
    with open(target_file, encoding="utf-8") as f:
        j = __import__("json").load(f)
    target = pd.DataFrame({"outputs": [x["output"] for x in j["outputs"]]})
else:
    try:
        target = pd.read_csv(target_file)
    except Exception:
        target = pd.read_csv(target_file, engine="python")

if args.correct:
    decode_selfies = "biot5" in args.name.lower()
    if decode_selfies:
        try:
            import selfies
        except ImportError:
            selfies = None
            decode_selfies = False
    corrected = []
    for raw in target["outputs"].astype(str):
        s = correct_text(raw)
        if decode_selfies and selfies is not None:
            try:
                s = selfies.decoder(s)
            except Exception:
                s = "None"
        corrected.append(s)
    target = pd.DataFrame({"outputs": corrected})

if args.benchmark == "open_generation":
    if args.task == "MolCustom":
        if args.subtask == "AtomNum":
            atom_type = ['carbon', 'oxygen', 'nitrogen', 'sulfur', 'fluorine', 'chlorine', 'bromine', 'iodine', 'phosphorus', 'boron', 'silicon', 'selenium', 'tellurium', 'arsenic', 'antimony', 'bismuth', 'polonium']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                if mol_prop(target["outputs"][idx], "validity"):
                    valid_molecules.append(target["outputs"][idx])
                    flag = 1
                    for atom in atom_type:
                        if mol_prop(target["outputs"][idx], "num_" + atom) != int(row[atom]):
                            flag = 0
                            break
                    flags.append(flag)
                else:
                    flags.append(0)
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
                print("Novelty: ", sum(novelties) / len(novelties))

        elif args.subtask == "FunctionalGroup":
            functional_groups = ['benzene rings', 'hydroxyl', 'anhydride', 'aldehyde', 'ketone', 'carboxyl', 'ester', 'amide', 'amine', 'nitro', 'halo', 'nitrile', 'thiol', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone', 'borane']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                if mol_prop(target["outputs"][idx], "validity"):
                    valid_molecules.append(target["outputs"][idx])
                    flag = 1
                    for group in functional_groups:
                        if group == "benzene rings":
                            if mol_prop(target["outputs"][idx], "num_benzene_ring") != int(row[group]):
                                flag = 0
                                break
                        else:
                            if mol_prop(target["outputs"][idx], "num_" + group) != int(row[group]):
                                flag = 0
                                break
                    flags.append(flag)
                else:
                    flags.append(0)
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
                print("Novelty: ", sum(novelties) / len(novelties))

        elif args.subtask == "BondNum":
            bonds_type = ['single', 'double', 'triple', 'rotatable', 'aromatic']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                if mol_prop(target["outputs"][idx], "validity"):
                    valid_molecules.append(target["outputs"][idx])
                    flag = 1
                    for bond in bonds_type:
                        if bond == "rotatable":
                            if int(row[bond]) == 0:
                                continue
                            elif mol_prop(target["outputs"][idx], "rot_bonds") != int(row[bond]):
                                flag = 0
                                break
                        else:
                            if int(row[bond]) == 0:
                                continue
                            elif mol_prop(target["outputs"][idx], "num_" + bond + "_bonds") != int(row[bond]):
                                flag = 0
                                break
                    flags.append(flag)
                else:
                    flags.append(0)
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
                print("Novelty: ", sum(novelties) / len(novelties))

    elif args.task == "MolEdit":
        if args.subtask == "AddComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                raw, group = row["molecule"], row["added_group"]
                if group == "benzene ring":
                    group = "benzene_ring"
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    successed.append(1 if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) + 1 else 0)
                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / data_len)
        elif args.subtask == "DelComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                raw, group = row["molecule"], row["removed_group"]
                if group == "benzene ring":
                    group = "benzene_ring"
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    successed.append(1 if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) - 1 else 0)
                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / data_len)

        elif args.subtask == "SubComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                raw = row["molecule"]
                added_group = row["added_group"]
                removed_group = row["removed_group"]
                if added_group == "benzene ring":
                    added_group = "benzene_ring"
                if removed_group == "benzene ring":
                    removed_group = "benzene_ring"
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    ok = (mol_prop(target_mol, "num_" + removed_group) == mol_prop(raw, "num_" + removed_group) - 1
                          and mol_prop(target_mol, "num_" + added_group) == mol_prop(raw, "num_" + added_group) + 1)
                    successed.append(1 if ok else 0)
                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / data_len)
            

    elif args.task == "MolOpt":
        if args.subtask == "LogP":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                raw, instruction = row["molecule"], row["Instruction"]
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    similarities.append(calculate_similarity(raw, target_mol))
                    lower_ok = "lower" in instruction or "decrease" in instruction
                    if lower_ok and mol_prop(target_mol, "logP") < mol_prop(raw, "logP"):
                        successed.append(1)
                    elif not lower_ok and mol_prop(target_mol, "logP") > mol_prop(raw, "logP"):
                        successed.append(1)
                    else:
                        successed.append(0)
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / data_len)

        elif args.subtask == "MR":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                raw, instruction = row["molecule"], row["Instruction"]
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    similarities.append(calculate_similarity(raw, target_mol))
                    lower_ok = "lower" in instruction or "decrease" in instruction
                    if lower_ok and mol_prop(target_mol, "MR") < mol_prop(raw, "MR"):
                        successed.append(1)
                    elif not lower_ok and mol_prop(target_mol, "MR") > mol_prop(raw, "MR"):
                        successed.append(1)
                    else:
                        successed.append(0)
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / data_len)
        elif args.subtask == "QED":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(data_len)):
                row = data[idx]
                raw, instruction = row["molecule"], row["Instruction"]
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)
                    similarities.append(calculate_similarity(raw, target_mol))
                    lower_ok = "lower" in instruction or "decrease" in instruction
                    if lower_ok and mol_prop(target_mol, "qed") < mol_prop(raw, "qed"):
                        successed.append(1)
                    elif not lower_ok and mol_prop(target_mol, "qed") > mol_prop(raw, "qed"):
                        successed.append(1)
                    else:
                        successed.append(0)
                else:
                    successed.append(0)
            print("Success Rate:", sum(successed) / len(successed))
            print("Similarity:", sum(similarities) / len(similarities))
            print("Validty:", len(valid_molecules) / data_len)
elif args.benchmark == "targeted_generation":
    pass
else:
    raise ValueError("Invalid Benchmark Type")