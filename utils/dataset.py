"""
S²-Bench and OpenMolIns data loading via Hugging Face datasets.
Supports loading from Hugging Face Hub or local CSV.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import selfies
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset

# ---------------------------------------------------------------------------
# Constants: prompt templates
# ---------------------------------------------------------------------------
SYSTEM_HEAD = (
    "You are working as an assistant of a chemist user. Please follow the instruction of the chemist "
    "and generate a molecule that satisfies the requirements of the chemist user. You could think step by step, "
    "but your final response should be a SMILES string. For example, 'Molecule: [SMILES STRING]'."
)
SYSTEM_HEAD_JSON = (
    "You are working as an assistant of a chemist user. Please follow the instruction of the chemist "
    "and generate a molecule that satisfies the requirements of the chemist user. Your final response should be "
    "a JSON object with the key 'molecule' and the value as a SMILES string. "
    "For example, {\"molecule\": \"[SMILES_STRING]\"}."
)

# Benchmark config name: (task, subtask) -> HF config or filename
BENCHMARK_CONFIG_MAP = {
    ("MolCustom", "AtomNum"): "MolCustom_AtomNum",
    ("MolCustom", "BondNum"): "MolCustom_BondNum",
    ("MolCustom", "FunctionalGroup"): "MolCustom_FunctionalGroup",
    ("MolEdit", "AddComponent"): "MolEdit_AddComponent",
    ("MolEdit", "DelComponent"): "MolEdit_DelComponent",
    ("MolEdit", "SubComponent"): "MolEdit_SubComponent",
    ("MolOpt", "LogP"): "MolOpt_LogP",
    ("MolOpt", "MR"): "MolOpt_MR",
    ("MolOpt", "QED"): "MolOpt_QED",
}

HF_BENCHMARK_REPO = "phenixace/S2-TOMG-Bench"
HF_BENCHMARK_REPO_MINI = "phenixace/S2-TOMG-Bench-mini"
HF_OPENMOLINS_REPO_PREFIX = "phenixace/OpenMolIns-"


def _resolve_benchmark_repo(hf_benchmark_repo: Optional[str]) -> str:
    """Return HF repo name: None or 'full' -> full bench, 'mini' -> mini bench."""
    if hf_benchmark_repo is None or hf_benchmark_repo == "full":
        return HF_BENCHMARK_REPO
    if hf_benchmark_repo == "mini":
        return HF_BENCHMARK_REPO_MINI
    return hf_benchmark_repo


def _get_benchmark_config_name(task: str, subtask: str) -> str:
    key = (task, subtask)
    if key not in BENCHMARK_CONFIG_MAP:
        return f"{task}_{subtask}"
    return BENCHMARK_CONFIG_MAP[key]


def load_benchmark_dataset(
    task: str,
    subtask: str,
    benchmark: str = "open_generation",
    split: str = "test",
    use_hf: bool = True,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_benchmark_repo: Optional[str] = None,
) -> Dataset:
    """
    Load S²-Bench benchmark, return Hugging Face Dataset.
    Load from Hub phenixace/S2-TOMG-Bench (or phenixace/S2-TOMG-Bench-mini) first, else from local CSV.
    hf_benchmark_repo: None or 'full' -> full; 'mini' -> mini; or full repo path.
    """
    config_name = _get_benchmark_config_name(task, subtask)
    repo = _resolve_benchmark_repo(hf_benchmark_repo)

    if use_hf:
        try:
            ds = load_dataset(
                repo,
                config_name,
                split=split,
                cache_dir=cache_dir,
            )
            return ds
        except Exception as e:
            if data_dir is None:
                raise RuntimeError(
                    f"Failed to load {repo} from Hugging Face: {e}. "
                    "Set data_dir to load from local CSV."
                ) from e

    # Local CSV: support two path layouts
    if data_dir is None:
        data_dir = "."
    # 1) data_dir/benchmarks/open_generation/MolCustom/AtomNum/test.csv
    path1 = os.path.join(data_dir, "benchmarks", benchmark, task, subtask, "test.csv")
    # 2) data_dir/full/MolCustom_AtomNum.csv
    path2 = os.path.join(data_dir, "full", f"{config_name}.csv")

    for path in [path1, path2]:
        if os.path.isfile(path):
            return load_dataset("csv", data_files={split: path}, split=split)

    raise FileNotFoundError(
        f"Benchmark data not found. Tried: {path1}, {path2}. "
        f"Or load from HF: load_dataset('{repo}', '{config_name}')"
    )


def load_openmolins_dataset(
    data_scale: str,
    split: str = "train",
    use_hf: bool = True,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    加载 OpenMolIns 指令微调数据。
    Load from Hub phenixace/OpenMolIns-{scale} first, else from local train.csv.
    """
    repo = f"{HF_OPENMOLINS_REPO_PREFIX}{data_scale}"

    if use_hf:
        try:
            return load_dataset(repo, split=split, trust_remote_code=True, cache_dir=cache_dir)
        except Exception as e:
            if data_dir is None:
                raise RuntimeError(
                    f"Failed to load {repo} from Hugging Face: {e}. Set data_dir for local load."
                ) from e

    # Local: data_dir/OpenMolIns/small/train.csv or data_dir/small/train.csv
    for base in [os.path.join(data_dir or ".", "OpenMolIns", data_scale), os.path.join(data_dir or ".", data_scale)]:
        path = os.path.join(base, "train.csv")
        if os.path.isfile(path):
            return load_dataset("csv", data_files={split: path}, split=split)

    raise FileNotFoundError(f"OpenMolIns data not found: {data_scale}. Tried data_dir/OpenMolIns/{data_scale}/train.csv")


# ---------------------------------------------------------------------------
# Dataset wrappers compatible with existing scripts (same __getitem__ behavior)
# ---------------------------------------------------------------------------


class OMGDataset(TorchDataset):
    """
    Open-domain molecule generation benchmark. Loaded via Hugging Face datasets; __getitem__ returns chat messages.
    """

    def __init__(
        self,
        task: str,
        subtask: str,
        json_check: bool = False,
        use_selfies: bool = False,
        benchmark: str = "open_generation",
        use_hf: bool = True,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_benchmark_repo: Optional[str] = None,
    ):
        self._hf_ds = load_benchmark_dataset(
            task, subtask, benchmark=benchmark, use_hf=use_hf, data_dir=data_dir, cache_dir=cache_dir,
            hf_benchmark_repo=hf_benchmark_repo,
        )
        self.instructions = list(self._hf_ds["Instruction"])
        if use_selfies and task in ("MolEdit", "MolOpt"):
            molecules = self._hf_ds["molecule"]
            self.instructions = []
            for i, inst in enumerate(self._hf_ds["Instruction"]):
                mol = molecules[i]
                try:
                    mol_selfies = selfies.encoder(mol)
                except Exception:
                    mol_selfies = mol
                self.instructions.append(inst.replace(mol, mol_selfies))
        self.json_check = json_check

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int) -> list[dict[str, str]]:
        query = self.instructions[idx]
        system = SYSTEM_HEAD_JSON if self.json_check else SYSTEM_HEAD
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]


class OMGInsTDataset(TorchDataset):
    """Benchmark instruction set with extra constraints; loaded via datasets."""

    ADDITIONAL_REQUIREMENTS = [
        "keep the logP value below 3",
        "keep the logP value above 3",
        "keep the molecular weight below 300",
        "keep the molecular weight above 300",
        "keep the ring count below 3",
        "keep the ring count above 3",
        "keep the qed value above 0.5",
        "keep the qed value below 0.5",
    ]

    def __init__(
        self,
        task: str,
        subtask: str,
        benchmark: str = "open_generation",
        use_hf: bool = True,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_benchmark_repo: Optional[str] = None,
    ):
        self._hf_ds = load_benchmark_dataset(
            task, subtask, benchmark=benchmark, use_hf=use_hf, data_dir=data_dir, cache_dir=cache_dir,
            hf_benchmark_repo=hf_benchmark_repo,
        )
        self.data = []
        self.targets = []
        for inst in self._hf_ds["Instruction"]:
            req = random.choice(self.ADDITIONAL_REQUIREMENTS)
            self.data.append("## User: " + (inst[:-1] if inst.endswith(".") else inst) + " and " + req + ".\n## Assistant: ")
            self.targets.append("")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.data[idx], self.targets[idx]


class TMGDataset(TorchDataset):
    """Targeted molecule generation benchmark; same interface as OMGDataset."""

    def __init__(
        self,
        task: str,
        subtask: str,
        json_check: bool = False,
        benchmark: str = "targeted_generation",
        use_hf: bool = True,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_benchmark_repo: Optional[str] = None,
    ):
        self._hf_ds = load_benchmark_dataset(
            task, subtask, benchmark=benchmark, use_hf=use_hf, data_dir=data_dir, cache_dir=cache_dir,
            hf_benchmark_repo=hf_benchmark_repo,
        )
        self.instructions = list(self._hf_ds["Instruction"])
        self.json_check = json_check

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int) -> list[dict[str, str]]:
        query = self.instructions[idx]
        system = SYSTEM_HEAD_JSON if self.json_check else SYSTEM_HEAD
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]


class InsTDataset(TorchDataset):
    """
    OpenMolIns instruction-tuning dataset. Loaded via Hugging Face datasets;
    exposes .data (input prefix) and .targets (full target sequence) for Trainer compatibility.
    """

    def __init__(
        self,
        data_scale: str,
        eos_token: str,
        specific_task: Optional[str] = None,
        special_token: bool = False,
        use_hf: bool = True,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        hf_ds = load_openmolins_dataset(
            data_scale, split="train", use_hf=use_hf, data_dir=data_dir, cache_dir=cache_dir
        )
        self.tasks = list(hf_ds["SubTask"])
        self.instructions = list(hf_ds["Instruction"])
        self.molecules = list(hf_ds["molecule"])
        self.data = []
        self.targets = []
        for i in range(len(self.instructions)):
            if specific_task is not None and self.tasks[i] != specific_task:
                continue
            prefix = "## User: " + self.instructions[i] + "\n## Assistant: "
            if special_token:
                gt = prefix + "[START_I_SMILES]{}[END_I_SMILES]".format(self.molecules[i]) + eos_token
            else:
                gt = prefix + self.molecules[i] + eos_token
            self.data.append(prefix)
            self.targets.append(gt)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.data[idx], self.targets[idx]


class SourceDataset(TorchDataset):
    """Generic (data, targets) wrapper with optional transform."""

    def __init__(self, data: list, targets: list, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample, target = self.data[idx], self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target


# ---------------------------------------------------------------------------
# 便捷 API：直接返回 HF Dataset（供 evaluate / 统计用）
# ---------------------------------------------------------------------------


def get_benchmark_hf_dataset(
    task: str,
    subtask: str,
    benchmark: str = "open_generation",
    use_hf: bool = True,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_benchmark_repo: Optional[str] = None,
) -> Dataset:
    """直接返回 benchmark 的 Hugging Face Dataset（用于 evaluate.py 等）。"""
    return load_benchmark_dataset(
        task, subtask, benchmark=benchmark, use_hf=use_hf, data_dir=data_dir, cache_dir=cache_dir,
        hf_benchmark_repo=hf_benchmark_repo,
    )


def get_openmolins_hf_dataset(
    data_scale: str,
    split: str = "train",
    use_hf: bool = True,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Return OpenMolIns as Hugging Face Dataset."""
    return load_openmolins_dataset(
        data_scale, split=split, use_hf=use_hf, data_dir=data_dir, cache_dir=cache_dir
    )


if __name__ == "__main__":
    # Local test: use parent dir for full/ or OpenMolIns
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(root)
    data_dir = parent  # tomg-hf

    print("Testing OMGDataset (local full/)...")
    ds = OMGDataset("MolCustom", "AtomNum", data_dir=data_dir, use_hf=False)
    print("len:", len(ds))
    print("sample:", ds[0])

    print("\nTesting InsTDataset (local OpenMolIns/small)...")
    inst_ds = InsTDataset("small", "<|end_of_text|>", data_dir=data_dir, use_hf=False)
    print("len:", len(inst_ds))
    print("sample raw:", inst_ds.data[0][:80], "...")
    print("sample gt:", inst_ds.targets[0][:80], "...")
