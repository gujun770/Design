"""

✅ 优化 1: 降低 QED 权重 (3.0 -> 1.0)，确保 Vina 结合能是第一优先级
✅ 优化 2: SA 改为连续评分，越容易合成(SA低)，总分越有优势
✅ 策略: Vina (主) + QED (辅) + SA (连续惩罚)
"""

import os
import sys
import random
import string
import shutil
import tempfile
import subprocess
import logging
import pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
from rdkit import RDLogger
import selfies as sf

RDLogger.DisableLog('rdApp.*')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 🔧 配置区域 =================
VINA_PATH = r"D:\vina.exe"
OBABEL_PATH = "obabel"

TARGETS = {
    "DRD2": {
        "receptor_paths": ["proteins/6CM4_clean.pdbqt", "docking_results_stoned_v4/proteins/6CM4_clean.pdbqt"],
        "center": {'x': 9.93, 'y': 5.67, 'z': -12.18},
        "box_size": {'x': 20, 'y': 20, 'z': 20},
        # ✅ DRD2 双子星 (含氟/含氧)
        "manual_seeds": [
            "C1=CCCC=NC=C1C=CC2=CC=C2F",
            "C1CCC=NC=C1C=CC2=CC=C2O"
        ]
    },
    "JAK2": {
        "receptor_paths": ["proteins/3UGC_clean.pdbqt", "docking_results_stoned_v4/proteins/3UGC_clean.pdbqt"],
        "center": {'x': 21.883, 'y': -18.273, 'z': -14.606},
        "box_size": {'x': 20, 'y': 20, 'z': 20},
        # ✅ JAK2 三剑客
        "manual_seeds": [
            "C1C=CCC=CCC=C2[NH1]C2CCC1[NH1]",
            "C1C=CCC=CC[NH1]C=C2[NH1]C2CCC1[NH1]",
            "C1CNCC=CCC=C2[NH1]C2CCOC1"
        ]
    }
}

GENERATIONS = 50
POP_SIZE = 50
TOP_K = 15

# ⚖️ 权重调整 (关键修改)
# 1.0 意味着: QED 提升 0.1 等价于 Vina 降低 0.1 (比较公平)
WEIGHT_QED = 1.0
# SA 系数: SA 每增加 1 (变难)，总分增加 0.2 (变差)
WEIGHT_SA = 0.2

# ================= 🛡️ 1. 硬过滤器 =================
def check_hard_filters(mol):
    if mol is None: return False

    # 必须含氮
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    if 'N' not in atoms: return False

    # 反作弊 (O=C=C 等)
    bad_patterns = [
        "[O]-[O]", "[N]-[N]", "[N]-[O]", "[O]-[N]", "[S]-[N]", "[N]-[S]",
        "[C]=[C]=[O]", "[C]=[C]=[C]"
    ]
    for pat in bad_patterns:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pat)): return False

    return True

# ================= 🧰 2. 连续 SA 评分系统 =================
def calc_real_sa_score(mol):
    """
    模拟 SA Score (1=最易, 10=最难)
    基于分子特征的加权计算
    """
    score = 1.0 # 基础分

    # 1. 环的复杂度 (Ring Complexity)
    # 桥环/螺环非常难做，罚分重
    num_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    score += num_bridge * 1.5
    score += num_spiro * 1.0

    # 大环 (>8) 难做
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        if len(ring) > 8: score += 2.0
        elif len(ring) > 6: score += 0.5 # 7,8元环稍难

    # 2. 手性中心 (Stereo Centers)
    # 手性越多，分离越难
    num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    if num_stereo > 1: score += (num_stereo - 1) * 0.5

    # 3. 分子大小 (Size)
    # 太大的分子通常合成步骤多
    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy > 25: score += (num_heavy - 25) * 0.1

    # 4. 杂原子比例 (Heteroatoms)
    # 并不是越多越好，适量即可

    # 限制范围 1-10
    return min(max(score, 1.0), 10.0)

# ================= 🧬 进化核心 =================
def get_safe_alphabet():
    base = ["[C]", "[N]", "[O]", "[F]", "[=C]", "[=N]", "[=O]",
            "[Branch1]", "[Branch2]", "[Ring1]", "[Ring2]"]
    extra = ["[S]", "[Cl]", "[Br]", "[NH1]"]
    return base + extra
ALPHABET = get_safe_alphabet()

def run_vina_scoring(smiles, receptor, center, box):
    if not smiles: return 0.0
    tmp_dir = tempfile.mkdtemp()
    unique_id = "".join(random.choices(string.ascii_lowercase, k=8))
    ligand_pdbqt = os.path.join(tmp_dir, f"lig_{unique_id}.pdbqt")
    out_pdbqt = os.path.join(tmp_dir, f"out_{unique_id}.pdbqt")

    try:
        if not os.path.exists(VINA_PATH): return 0.0
        cmd_conv = [OBABEL_PATH, "-:" + smiles, "-O", ligand_pdbqt, "--gen3d", "-p", "7.4", "--partialcharge", "gasteiger"]
        subprocess.run(cmd_conv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)

        if not os.path.exists(ligand_pdbqt):
            shutil.rmtree(tmp_dir); return 0.0

        cmd_vina = [
            VINA_PATH, "--receptor", receptor, "--ligand", ligand_pdbqt,
            "--center_x", str(center['x']), "--center_y", str(center['y']), "--center_z", str(center['z']),
            "--size_x", str(box['x']), "--size_y", str(box['y']), "--size_z", str(box['z']),
            "--cpu", "1", "--exhaustiveness", "4", "--out", out_pdbqt
        ]
        result = subprocess.run(cmd_vina, capture_output=True, text=True, timeout=30)

        best_affinity = 0.0
        for line in result.stdout.split('\n'):
            if line.strip().startswith('1'):
                try: best_affinity = float(line.split()[1])
                except: pass
                break
        shutil.rmtree(tmp_dir)
        return best_affinity
    except:
        try: shutil.rmtree(tmp_dir)
        except: pass
        return 0.0

def mutate_selfies(selfies_str):
    try: chars = list(sf.split_selfies(selfies_str))
    except: return selfies_str
    if not chars: return selfies_str
    choice = random.choices(["insert", "replace", "delete"], weights=[0.2, 0.6, 0.2], k=1)[0]
    if choice == "insert": chars.insert(random.randint(0, len(chars)), random.choice(ALPHABET))
    elif choice == "replace": chars[random.randint(0, len(chars)-1)] = random.choice(ALPHABET)
    elif choice == "delete" and len(chars) > 1: chars.pop(random.randint(0, len(chars)-1))
    return "".join(chars)

def evaluate_worker(args):
    smiles, receptor, center, box_size = args
    mol = Chem.MolFromSmiles(smiles)

    # 硬过滤器失败
    if not check_hard_filters(mol):
        return (smiles, 10.0, 10.0, 0.0, 10.0) # SA给10(最难)

    vina_score = run_vina_scoring(smiles, receptor, center, box_size)
    if vina_score >= 0: return (smiles, 10.0, 10.0, 0.0, 10.0)

    # 计算指标
    qed_score = QED.qed(mol)
    sa_score = calc_real_sa_score(mol) # 1.0 ~ 10.0

    # 🔥 修正后的打分公式 🔥
    # Final = Vina - (QED * 1.0) + (SA * 0.2)
    # 例子: -9.0 - 0.7 + 0.4 = -9.3 (越低越好)
    final_score = vina_score - (qed_score * WEIGHT_QED) + (sa_score * WEIGHT_SA)

    return (smiles, final_score, vina_score, qed_score, sa_score)

# ================= 🚀 主流程 =================
def run_evolution(target_name):
    # 1. 准备受体
    receptor = None
    if TARGETS[target_name]["receptor_paths"]:
        for p in TARGETS[target_name]["receptor_paths"]:
            if os.path.exists(p): receptor = p; break
    if not receptor: logger.error(f"❌ 找不到受体"); return None

    conf = TARGETS[target_name]
    logger.info(f"🚀 启动 {target_name} 完美进化 (修正权重)...")

    # 2. 准备种子
    valid_seeds = conf["manual_seeds"]
    population_selfies = []
    for s in valid_seeds:
        try: population_selfies.append(sf.encoder(s))
        except: pass

    while len(population_selfies) < POP_SIZE:
        base_seed = random.choice(valid_seeds)
        try: population_selfies.append(mutate_selfies(sf.encoder(base_seed)))
        except: pass

    # 3. 进化循环
    pool = Pool(processes=4)
    global_memory = {}

    for gen in range(GENERATIONS):
        pop_smiles = [sf.decoder(s) for s in population_selfies]
        pop_smiles = [s for s in pop_smiles if s]

        tasks = []
        for s in pop_smiles:
            if s in global_memory: tasks.append(None)
            else: tasks.append((s, receptor, conf['center'], conf['box_size']))

        real_tasks = [t for t in tasks if t is not None]
        if real_tasks: results = pool.map(evaluate_worker, real_tasks)

        current_data = []
        res_idx = 0

        for i, task in enumerate(tasks):
            s = pop_smiles[i]
            if task is None:
                metrics = global_memory[s]
            else:
                _, final, vina, qed, sa = results[res_idx]
                metrics = (final, vina, qed, sa)
                global_memory[s] = metrics
                res_idx += 1
            current_data.append(metrics)

        # 打印 Best
        final_scores = [x[0] for x in current_data]
        best_idx = np.argmin(final_scores)
        best_metrics = current_data[best_idx]
        best_smi = pop_smiles[best_idx]

        # 打印真实 SA 值
        logger.info(f"   Gen {gen+1:02d}: Final={best_metrics[0]:.2f} | Vina={best_metrics[1]:.2f} | QED={best_metrics[2]:.2f} | SA={best_metrics[3]:.1f} | {best_smi}")

        # 繁殖
        indices = np.argsort(final_scores)[:TOP_K]
        elites = [population_selfies[i] for i in indices]

        new_pop = list(elites)
        while len(new_pop) < POP_SIZE:
            new_pop.append(mutate_selfies(random.choice(elites)))
        population_selfies = new_pop

    pool.close(); pool.join()

    valid_mols = [(s, m[1]) for s, m in global_memory.items() if m[1] < -5.0]
    valid_mols.sort(key=lambda x: x[1])
    return valid_mols[:50]

def main():
    out_dir = Path("results/exported_molecules_perfect")
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in ["DRD2", "JAK2"]:
        res = run_evolution(target)
        if not res: continue

        smiles = [x[0] for x in res]
        scores = [x[1] for x in res]
        dta = [-x for x in scores]

        with open(out_dir / f"{target}_generated.pkl", "wb") as f:
            pickle.dump({"smiles": smiles, "dta_history": dta, "target": target}, f)
        logger.info(f"✅ {target} 完美优化完成! Final Best Vina: {scores[0]:.2f}")

if __name__ == "__main__":
    main()