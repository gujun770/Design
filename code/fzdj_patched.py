import subprocess
import tempfile
import os
from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm

# ================= 配置区 =================
VINA_PATH = r"D:\vina.exe"

# 你的蛋白质和对接结果目录
PROTEIN_DIR = Path('docking_results_perfect/proteins')
DOCKED_DIR = Path('docking_results_perfect/docked')

TARGETS = {
    'JAK2': {
        'filename': '3UGC_clean.pdbqt',
        'center': [21.883, -18.273, -14.606],
        'size': [30.0, 30.0, 30.0]
    },
    'DRD2': {
        'filename': '6CM4_clean.pdbqt',
        'center': [14.129, 10.921, 3.314],
        'size': [30.0, 30.0, 30.0]
    }
}


# ==========================================

def extract_first_model(multi_model_file, output_file):
    """提取 MODEL 1"""
    with open(multi_model_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    first_model_lines = []
    in_model_1 = False
    has_model_tags = any(line.startswith("MODEL") for line in lines)

    if not has_model_tags:
        first_model_lines = lines
    else:
        for line in lines:
            if line.startswith("MODEL 1") or (line.startswith("MODEL") and not in_model_1 and not first_model_lines):
                in_model_1 = True
                continue
            elif line.startswith("ENDMDL") and in_model_1:
                break
            elif in_model_1:
                first_model_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(first_model_lines)


def run_vinardo_score_only(protein_pdbqt, docked_ligand_pdbqt, center, size):
    """调用 Vina 1.2+ 进行打分，附带网格参数"""
    prot_abs_path = str(protein_pdbqt.resolve())

    fd, temp_ligand_path = tempfile.mkstemp(suffix='.pdbqt', text=True)
    os.close(fd)

    try:
        extract_first_model(docked_ligand_pdbqt, temp_ligand_path)

        cmd = [
            VINA_PATH,
            '--receptor', prot_abs_path,
            '--ligand', temp_ligand_path,
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', str(size[0]),
            '--size_y', str(size[1]),
            '--size_z', str(size[2]),
            '--scoring', 'vinardo',
            '--score_only'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"\n⚠️ {docked_ligand_pdbqt.name} Vina 报错：\n{result.stderr.strip()}")
            return None

        # 👇 终极修改：完美匹配 "Estimated Free Energy of Binding : -3.863"
        for line in result.stdout.split('\n'):
            match = re.search(r'(?:Affinity|Score|Estimated Free Energy).*?:\s*([-\d.]+)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))

        print(f"\n[调试探针] 未能在 {docked_ligand_pdbqt.name} 中找到分数。Vina 的真实输出如下：")
        print("-" * 50)
        print(result.stdout.strip())
        print("-" * 50)
        return None

    except Exception as e:
        print(f"\n❌ Python执行异常 {docked_ligand_pdbqt.name}: {e}")
        return None
    finally:
        if os.path.exists(temp_ligand_path):
            os.remove(temp_ligand_path)


def main():
    if not Path(VINA_PATH).exists():
        print(f"❌ 找不到 Vina 程序: {VINA_PATH}")
        return

    print("🚀 开始进行 Vinardo 正交重新打分 (基于 Vina 1.2+ 提取单姿势并添加网格)...")

    all_results = []

    for target, config in TARGETS.items():
        protein_path = PROTEIN_DIR / config['filename']
        if not protein_path.exists():
            print(f"⚠️ 找不到蛋白质文件 {protein_path}，跳过 {target}")
            continue

        target_docked_dir = DOCKED_DIR / target
        if not target_docked_dir.exists():
            print(f"⚠️ 找不到对接结果目录 {target_docked_dir}，跳过 {target}")
            continue

        out_files = list(target_docked_dir.glob("*.pdbqt"))
        if not out_files:
            continue

        print(f"\n🎯 正在处理 {target} (共 {len(out_files)} 个分子的最优构象)...")

        for out_file in tqdm(out_files):
            mol_id_match = re.search(r'mol_(\d+)', out_file.name)
            mol_id = int(mol_id_match.group(1)) if mol_id_match else "Unknown"

            vinardo_score = run_vinardo_score_only(protein_path, out_file, config['center'], config['size'])

            if vinardo_score is not None:
                all_results.append({
                    'Target': target,
                    'Mol_ID': mol_id,
                    'Vinardo_Score': vinardo_score,
                    'File_Name': out_file.name
                })

    if all_results:
        df_rescore = pd.DataFrame(all_results)

        for target in TARGETS.keys():
            vina_csv = Path(f'docking_results_perfect/{target}_docking_results.csv')
            df_target_rescore = df_rescore[df_rescore['Target'] == target]

            if vina_csv.exists() and not df_target_rescore.empty:
                df_vina = pd.read_csv(vina_csv)
                df_merged = pd.merge(df_vina, df_target_rescore[['Mol_ID', 'Vinardo_Score']],
                                     left_on='mol_id', right_on='Mol_ID', how='left')

                if 'Mol_ID' in df_merged.columns:
                    df_merged = df_merged.drop(columns=['Mol_ID'])

                merged_csv_path = Path(f'docking_results_perfect/{target}_docking_rescore_results.csv')
                df_merged.to_csv(merged_csv_path, index=False)
                print(f"✅ {target} 的综合打分已保存至: {merged_csv_path}")
            elif not df_target_rescore.empty:
                standalone_csv_path = Path(f'docking_results_perfect/{target}_vinardo_only_results.csv')
                df_target_rescore.to_csv(standalone_csv_path, index=False)
                print(f"✅ {target} 的 Vinardo 打分已保存至: {standalone_csv_path}")
    else:
        print("\n⚠️ 未能成功提取任何分数。请查看上面的[调试探针]输出！")


if __name__ == "__main__":
    main()