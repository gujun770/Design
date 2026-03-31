import subprocess
from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm

# ================= 配置区 =================
# 直接使用你电脑上现有的 Vina 1.2.7+
VINA_PATH = r"D:\vina.exe"

# 你的蛋白质和对接结果目录
PROTEIN_DIR = Path('docking_results_perfect/proteins')
DOCKED_DIR = Path('docking_results_perfect/docked')

# 靶点配置 (对应你清理去水后的蛋白 pdbqt 文件)
TARGETS = {
    'JAK2': '3UGC_clean.pdbqt',
    'DRD2': '6CM4_clean.pdbqt'
}
# ==========================================

def run_vinardo_score_only(protein_pdbqt, docked_ligand_pdbqt):
    """调用 Vina 1.2+ 使用 Vinardo 评分函数对已有姿势进行重新打分"""

    cmd = [
        VINA_PATH,
        '--receptor', str(protein_pdbqt),
        '--ligand', str(docked_ligand_pdbqt),
        '--scoring', 'vinardo',  # 👈 关键点：指定使用 vinardo 经验打分函数
        '--score_only'  # 👈 关键点：仅打分，不改变原有的对接构象
    ]

    try:
        # 运行 Vina
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"⚠️ 运行警告: {docked_ligand_pdbqt.name}")
            return None

        # 解析 Vina --score_only 的输出日志寻找分数
        for line in result.stdout.split('\n'):
            if 'Affinity:' in line:
                # 匹配 Affinity: -8.12345 (kcal/mol)
                match = re.search(r'Affinity:\s*([-\d.]+)', line)
                if match:
                    return float(match.group(1))

    except Exception as e:
        print(f"❌ 运行异常 {docked_ligand_pdbqt.name}: {e}")
        return None


def main():
    if not Path(VINA_PATH).exists():
        print(f"❌ 找不到 Vina 程序: {VINA_PATH}")
        return

    print("🚀 开始进行 Vinardo 正交重新打分 (基于 Vina 1.2+)...")

    all_results = []

    for target, protein_filename in TARGETS.items():
        protein_path = PROTEIN_DIR / protein_filename
        if not protein_path.exists():
            print(f"⚠️ 找不到蛋白质文件 {protein_path}，跳过 {target}")
            continue

        target_docked_dir = DOCKED_DIR / target
        if not target_docked_dir.exists():
            print(f"⚠️ 找不到对接结果目录 {target_docked_dir}，跳过 {target}")
            continue

        # 👇 关键修正：将 *_out.pdbqt 改为了 *.pdbqt，适配最新的文件名格式
        out_files = list(target_docked_dir.glob("*.pdbqt"))
        if not out_files:
            print(f"⚠️ 在 {target_docked_dir} 下没有找到任何 .pdbqt 文件！")
            continue

        print(f"\n🎯 正在处理 {target} (共 {len(out_files)} 个分子的构象)...")

        for out_file in tqdm(out_files):
            # 从文件名提取分子 ID (例如: JAK2_mol_001.pdbqt)
            mol_id_match = re.search(r'mol_(\d+)', out_file.name)
            mol_id = int(mol_id_match.group(1)) if mol_id_match else "Unknown"

            vinardo_score = run_vinardo_score_only(protein_path, out_file)

            if vinardo_score is not None:
                all_results.append({
                    'Target': target,
                    'Mol_ID': mol_id,
                    'Vinardo_Score': vinardo_score,
                    'File_Name': out_file.name
                })

    # 合并并保存结果
    if all_results:
        df_rescore = pd.DataFrame(all_results)

        for target in TARGETS.keys():
            # 尝试读取原始的 Vina 结果文件，把新分数拼进去
            vina_csv = Path(f'docking_results_perfect/{target}_docking_results.csv')
            df_target_rescore = df_rescore[df_rescore['Target'] == target]

            if vina_csv.exists() and not df_target_rescore.empty:
                df_vina = pd.read_csv(vina_csv)

                # 合并数据
                df_merged = pd.merge(df_vina, df_target_rescore[['Mol_ID', 'Vinardo_Score']],
                                     left_on='mol_id', right_on='Mol_ID', how='left')

                # 丢弃多余的 Mol_ID 列
                if 'Mol_ID' in df_merged.columns:
                    df_merged = df_merged.drop(columns=['Mol_ID'])

                merged_csv_path = Path(f'docking_results_perfect/{target}_docking_rescore_results.csv')
                df_merged.to_csv(merged_csv_path, index=False)
                print(f"✅ {target} 的综合打分已保存至: {merged_csv_path}")
            elif not df_target_rescore.empty:
                # 如果没有原始结果文件，直接保存正交打分结果
                standalone_csv_path = Path(f'docking_results_perfect/{target}_vinardo_only_results.csv')
                df_target_rescore.to_csv(standalone_csv_path, index=False)
                print(f"✅ {target} 的 Vinardo 打分已保存至: {standalone_csv_path}")
    else:
        print("⚠️ 未能成功提取任何分数。")


if __name__ == "__main__":
    main()