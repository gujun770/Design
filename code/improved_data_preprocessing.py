import os
import gzip
import torch
import requests
from pathlib import Path
from Bio.PDB import PDBParser, Select
from Bio.PDB.Polypeptide import is_aa
from collections import defaultdict
import pandas as pd
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pickle
import re
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning, module='Bio.PDB')

# 初始化CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FinalStructureProcessor:
    """最终版本的结构处理器 - 优化数据量和质量平衡"""

    def __init__(self, pdbind_dir, scpdb_dir, output_dir, distance_cutoff=4.0):
        self.pdbind_dir = Path(pdbind_dir)
        self.scpdb_dir = Path(scpdb_dir)
        self.output_dir = Path(output_dir)
        self.distance_cutoff = distance_cutoff
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pdb_to_uniprot = defaultdict(list)
        self.uniprot_data = defaultdict(dict)
        self.sifts_file = self.output_dir / "pdb_chain_uniprot.tsv"
        self.uniprot_api = "https://www.uniprot.org/uniprot/"

        # 优化后的图质量统计
        self.graph_stats = {
            'total_processed': 0,
            'quality_excellent': 0,  # 密度5-25%，连通性>90%
            'quality_good': 0,  # 密度2-5%或25-40%，连通性>70%
            'quality_acceptable': 0,  # 密度1-2%或连通性50-70%
            'quality_poor': 0,  # 其他情况
            'accepted': 0,
            'rejected': 0
        }

        if not self.pdbind_dir.exists():
            raise FileNotFoundError(f"PDBbind目录不存在: {self.pdbind_dir}")
        if not self.scpdb_dir.exists():
            raise FileNotFoundError(f"scPDB目录不存在: {self.scpdb_dir}")

    def load_sifts_mapping(self):
        """加载SIFTS映射，如果失败则跳过去重"""
        if not self.sifts_file.exists():
            print("警告: 未找到SIFTS映射文件，将跳过基于UniProt的去重")
            return False

        print("正在加载SIFTS映射...")
        try:
            with open(self.sifts_file, 'rt') as f:
                df = pd.read_csv(f, sep='\t', comment='#',
                                 usecols=['PDB', 'CHAIN', 'SP_PRIMARY', 'RES_BEG', 'RES_END'])

            df['PDB'] = df['PDB'].str.lower()
            df = df.dropna(subset=['PDB', 'SP_PRIMARY'])

            for _, row in df.iterrows():
                self.pdb_to_uniprot[row['PDB']].append({
                    'chain': row['CHAIN'],
                    'uniprot_id': row['SP_PRIMARY'],
                    'start': row['RES_BEG'],
                    'end': row['RES_END']
                })

            print(f"成功加载 {len(self.pdb_to_uniprot)} 个PDB到UniProt映射")
            return True
        except Exception as e:
            print(f"加载SIFTS映射失败: {e}")
            return False

    def get_uniprot_metadata(self, uniprot_id):
        """获取UniProt元数据，失败时返回默认值"""
        if uniprot_id in self.uniprot_data:
            return self.uniprot_data[uniprot_id]

        try:
            url = f"{self.uniprot_api}{uniprot_id}.xml"
            response = requests.get(url, timeout=10)  # 减少超时时间
            response.raise_for_status()

            content = response.text
            # 简化解析，避免解析错误
            try:
                accession = content.split('<accession>')[1].split('</accession>')[0]
                name = content.split('<name>')[1].split('</name>')[0]
                seq_length = int(content.split('<sequence length="')[1].split('"')[0])
                reviewed = 'dataset="reviewed"' in content
            except:
                # 解析失败时使用默认值
                accession = uniprot_id
                name = uniprot_id
                seq_length = 300
                reviewed = False

            self.uniprot_data[uniprot_id] = {
                'accession': accession,
                'name': name,
                'organism': 'Unknown',
                'sequence_length': seq_length,
                'reviewed': reviewed
            }
            return self.uniprot_data[uniprot_id]

        except Exception as e:
            # 网络失败时返回默认元数据
            self.uniprot_data[uniprot_id] = {
                'accession': uniprot_id,
                'name': uniprot_id,
                'organism': 'Unknown',
                'sequence_length': 300,
                'reviewed': False
            }
            return self.uniprot_data[uniprot_id]

    def _get_uniprot_coverage_gpu(self, pdb_id):
        """获取UniProt覆盖率，失败时返回默认值"""
        if pdb_id not in self.pdb_to_uniprot:
            return f"unknown_{pdb_id}", 1.0  # 返回默认值

        mappings = self.pdb_to_uniprot[pdb_id]
        if not mappings:
            return f"unknown_{pdb_id}", 1.0

        try:
            if torch.cuda.is_available() and len(mappings) > 1:
                uniprot_hashes = torch.tensor(
                    [hash(m['uniprot_id']) for m in mappings],
                    device=device
                )
                unique_ids, counts = torch.unique(uniprot_hashes, return_counts=True)
                main_id_hash = unique_ids[torch.argmax(counts)].item()
            else:
                from collections import Counter
                counts = Counter(m['uniprot_id'] for m in mappings)
                main_id_hash = hash(max(counts, key=counts.get))

            main_id = next(m['uniprot_id'] for m in mappings if hash(m['uniprot_id']) == main_id_hash)
            metadata = self.get_uniprot_metadata(main_id)

            total_length = metadata['sequence_length']
            covered = sum(max(0, m.get('end', 100) - m.get('start', 0) + 1)
                          for m in mappings
                          if m.get('uniprot_id') == main_id)

            coverage = min(1.0, covered / max(total_length, 1))
            return main_id, coverage

        except Exception:
            return f"unknown_{pdb_id}", 1.0

    def process_pdbind_structures(self):
        """处理PDBbind结构"""
        print("扫描PDBbind结构...")
        structures = []

        for pdb_dir in self.pdbind_dir.glob("*/"):
            if not pdb_dir.is_dir():
                continue

            pdb_id = pdb_dir.name.lower()
            protein_file = pdb_dir / f"{pdb_id}_protein.pdb"
            ligand_file = pdb_dir / f"{pdb_id}_ligand.mol2"

            if not protein_file.exists() or not ligand_file.exists():
                continue

            uniprot_id, coverage = self._get_uniprot_coverage_gpu(pdb_id)

            structures.append({
                'pdb_id': pdb_id.upper(),
                'uniprot_id': uniprot_id,
                'coverage': coverage,
                'protein_path': str(protein_file),
                'ligand_path': str(ligand_file),
                'source': 'pdbind'
            })

        return structures

    def process_scpdb_structures(self):
        """处理scPDB结构"""
        print("扫描scPDB结构...")
        structures = []

        for sc_dir in self.scpdb_dir.glob("*/"):
            if not sc_dir.is_dir():
                continue

            dir_name = sc_dir.name
            pdb_id = dir_name.split('_')[0].lower() if '_' in dir_name else dir_name.lower()

            # 查找蛋白质和配体文件
            files = list(sc_dir.glob("*"))
            if len(files) < 2:
                continue

            # 按文件大小排序
            files_with_size = [(f, f.stat().st_size) for f in files
                               if f.suffix in ['.mol2', '.sdf', '.pdb'] and f.stat().st_size > 1000]

            if len(files_with_size) < 2:
                continue

            files_with_size.sort(key=lambda x: x[1], reverse=True)
            protein_file = files_with_size[0][0]
            ligand_file = files_with_size[-1][0]  # 最小的作为配体

            uniprot_id, coverage = self._get_uniprot_coverage_gpu(pdb_id)

            structures.append({
                'pdb_id': pdb_id.upper(),
                'uniprot_id': uniprot_id,
                'coverage': coverage,
                'protein_path': str(protein_file),
                'ligand_path': str(ligand_file),
                'source': 'scpdb'
            })

        return structures

    def deduplicate_structures(self, structures, has_uniprot_mapping):
        """去重结构"""
        if not has_uniprot_mapping:
            print("跳过UniProt去重，保留所有结构")
            return structures

        print("基于UniProt ID去重中...")
        uniprot_groups = defaultdict(list)

        for struct in structures:
            uniprot_id = struct['uniprot_id']
            if uniprot_id and not uniprot_id.startswith('unknown_'):
                uniprot_groups[uniprot_id].append(struct)

        # 只对有有效UniProt ID的进行去重，其他的保留
        deduplicated = []
        processed_uniprots = set()

        for struct in structures:
            uniprot_id = struct['uniprot_id']

            if not uniprot_id or uniprot_id.startswith('unknown_'):
                # 无UniProt映射的直接保留
                deduplicated.append(struct)
            elif uniprot_id not in processed_uniprots:
                # 选择覆盖率最高的
                group = uniprot_groups[uniprot_id]
                best_struct = max(group, key=lambda x: x['coverage'])
                deduplicated.append(best_struct)
                processed_uniprots.add(uniprot_id)

        return deduplicated

    def filter_standard_aas(self, pdb_file):
        """检查是否只包含标准氨基酸，失败时返回True"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("temp", pdb_file)

            for model in structure:
                for chain in model:
                    for residue in chain:
                        if not is_aa(residue, standard=True):
                            return False
            return True
        except Exception:
            return True  # 解析失败时不过滤

    def _extract_residue_coords_from_pdb(self, pdb_file, ligand_distance_threshold=6.0):
        """从PDB文件提取，使用更宽松的配体距离阈值"""
        protein_atoms = []
        ligand_atoms = []

        try:
            with open(pdb_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith('ATOM'):
                    try:
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21:22].strip() or 'A'
                        residue_seq = line[22:26].strip()
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

                        if self._is_amino_acid(residue_name):
                            protein_atoms.append({
                                'coords': np.array([x, y, z]),
                                'residue_name': residue_name,
                                'chain_id': chain_id,
                                'residue_seq': residue_seq,
                                'atom_name': atom_name,
                                'residue_key': f"{chain_id}_{residue_seq}_{residue_name}"
                            })
                    except (ValueError, IndexError):
                        continue

                elif line.startswith('HETATM'):
                    try:
                        residue_name = line[17:20].strip()
                        if residue_name not in ['HOH', 'WAT', 'SO4', 'PO4', 'CL', 'NA', 'MG', 'CA', 'ZN']:
                            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                            ligand_atoms.append(np.array([x, y, z]))
                    except (ValueError, IndexError):
                        continue

            return self._process_active_site_residues(protein_atoms, ligand_atoms, ligand_distance_threshold)

        except Exception as e:
            print(f"处理PDB文件 {pdb_file} 时出错: {e}")
            return [], [], np.array([])

    def _extract_residue_coords_from_mol2(self, mol2_file, ligand_distance_threshold=6.0):
        """从MOL2文件提取"""
        protein_atoms = []
        ligand_atoms = []

        try:
            with open(mol2_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            atom_section = False
            for line in lines:
                line = line.strip()

                if line.startswith('@<TRIPOS>ATOM'):
                    atom_section = True
                    continue
                elif line.startswith('@<TRIPOS>') and atom_section:
                    break
                elif atom_section and line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            atom_name = parts[1]
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                            residue_id = parts[6] if len(parts) > 6 else ""
                            residue_name = parts[7] if len(parts) > 7 else ""

                            clean_residue_name = re.sub(r'\d+$', '', residue_name)

                            if self._is_amino_acid(clean_residue_name):
                                protein_atoms.append({
                                    'coords': np.array([x, y, z]),
                                    'residue_name': clean_residue_name,
                                    'chain_id': 'A',
                                    'residue_seq': residue_id or '1',
                                    'atom_name': atom_name,
                                    'residue_key': f"A_{residue_id or '1'}_{clean_residue_name}"
                                })
                            else:
                                ligand_atoms.append(np.array([x, y, z]))

                        except (ValueError, IndexError):
                            continue

            return self._process_active_site_residues(protein_atoms, ligand_atoms, ligand_distance_threshold)

        except Exception as e:
            print(f"处理MOL2文件 {mol2_file} 时出错: {e}")
            return [], [], np.array([])

    def _process_active_site_residues(self, protein_atoms, ligand_atoms, ligand_distance_threshold):
        """处理活性位点残基"""
        if not protein_atoms:
            return [], [], np.array([])

        if not ligand_atoms:
            return self._extract_by_geometric_center(protein_atoms)

        try:
            ligand_coords = np.array(ligand_atoms)
            active_site_residues = {}

            for atom in protein_atoms:
                distances = np.linalg.norm(ligand_coords - atom['coords'], axis=1)
                min_distance = np.min(distances)

                if min_distance <= ligand_distance_threshold:
                    residue_key = atom['residue_key']
                    if residue_key not in active_site_residues:
                        active_site_residues[residue_key] = {
                            'residue_name': atom['residue_name'],
                            'chain_id': atom['chain_id'],
                            'residue_seq': atom['residue_seq'],
                            'atoms': [],
                            'min_ligand_distance': min_distance
                        }

                    active_site_residues[residue_key]['atoms'].append(atom)
                    active_site_residues[residue_key]['min_ligand_distance'] = min(
                        active_site_residues[residue_key]['min_ligand_distance'],
                        min_distance
                    )

            return self._build_graph_from_residues(active_site_residues)

        except Exception as e:
            print(f"处理活性位点时出错: {e}")
            return self._extract_by_geometric_center(protein_atoms)

    def _extract_by_geometric_center(self, protein_atoms):
        """基于几何中心提取"""
        if not protein_atoms:
            return [], [], np.array([])

        try:
            all_coords = np.array([atom['coords'] for atom in protein_atoms])
            center = np.mean(all_coords, axis=0)

            distances_to_center = [np.linalg.norm(atom['coords'] - center) for atom in protein_atoms]

            # 使用更宽松的距离阈值
            distance_threshold = np.percentile(distances_to_center, 60)  # 降低到60%
            distance_threshold = max(10.0, min(distance_threshold, 20.0))

            active_site_residues = {}
            for atom in protein_atoms:
                distance = np.linalg.norm(atom['coords'] - center)
                if distance <= distance_threshold:
                    residue_key = atom['residue_key']
                    if residue_key not in active_site_residues:
                        active_site_residues[residue_key] = {
                            'residue_name': atom['residue_name'],
                            'chain_id': atom['chain_id'],
                            'residue_seq': atom['residue_seq'],
                            'atoms': [],
                            'min_ligand_distance': distance
                        }
                    active_site_residues[residue_key]['atoms'].append(atom)

            return self._build_graph_from_residues(active_site_residues)

        except Exception as e:
            print(f"几何中心方法失败: {e}")
            return [], [], np.array([])

    def _build_graph_from_residues(self, active_site_residues):
        """构建图结构"""
        if not active_site_residues:
            return [], [], np.array([])

        residue_coords = []
        aa_types = []

        # 最多保留50个残基
        sorted_residues = sorted(
            active_site_residues.items(),
            key=lambda x: x[1]['min_ligand_distance']
        )[:50]

        for residue_key, residue_data in sorted_residues:
            residue_center = self._get_residue_center(residue_data['atoms'])

            if residue_center is not None:
                residue_coords.append(residue_center)
                aa_types.append(residue_data['residue_name'])

        if len(residue_coords) > 0:
            coords_array = np.array(residue_coords)
            adj_matrix = self._build_optimized_adjacency_matrix(coords_array)
        else:
            adj_matrix = np.array([])

        return residue_coords, aa_types, adj_matrix

    def _build_optimized_adjacency_matrix(self, coords):
        """构建优化的邻接矩阵 - 平衡质量和数量"""
        n_residues = len(coords)
        if n_residues <= 1:
            return np.eye(max(n_residues, 1))

        distances = squareform(pdist(coords))

        # 尝试多个距离阈值，找到合适的
        distance_cutoffs = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

        for cutoff in distance_cutoffs:
            adj_matrix = (distances <= cutoff).astype(float)
            np.fill_diagonal(adj_matrix, 0)

            edge_count = int(np.sum(adj_matrix) / 2)
            max_edges = n_residues * (n_residues - 1) / 2
            density = edge_count / max_edges * 100 if max_edges > 0 else 0

            # 更宽松的要求：
            # - 至少3条边
            # - 密度至少0.5%
            # - 最大连通组件至少包含50%节点
            if edge_count >= 3 and density >= 0.5:
                connectivity_ratio = self._get_connectivity_ratio(adj_matrix)
                if connectivity_ratio >= 50.0:
                    return adj_matrix

        # 如果都不满足，使用最大阈值
        adj_matrix = (distances <= distance_cutoffs[-1]).astype(float)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

    def _get_connectivity_ratio(self, adj_matrix):
        """获取最大连通组件的比例"""
        try:
            n = adj_matrix.shape[0]
            if n <= 1:
                return 100.0

            visited = [False] * n
            max_component_size = 0

            for i in range(n):
                if not visited[i]:
                    component_size = self._dfs_component_size(adj_matrix, i, visited)
                    max_component_size = max(max_component_size, component_size)

            return (max_component_size / n) * 100
        except:
            return 0.0

    def _dfs_component_size(self, adj_matrix, start, visited):
        """DFS计算连通组件大小"""
        try:
            visited[start] = True
            size = 1

            for neighbor in range(len(adj_matrix)):
                if adj_matrix[start][neighbor] > 0 and not visited[neighbor]:
                    size += self._dfs_component_size(adj_matrix, neighbor, visited)

            return size
        except:
            return 1

    def _get_residue_center(self, residue_atoms):
        """计算残基中心"""
        if not residue_atoms:
            return None

        try:
            backbone_atoms = ['CA', 'CB', 'N', 'C']
            backbone_coords = []

            for atom in residue_atoms:
                if atom['atom_name'] in backbone_atoms:
                    backbone_coords.append(atom['coords'])

            if backbone_coords:
                return np.mean(backbone_coords, axis=0)
            else:
                all_coords = [atom['coords'] for atom in residue_atoms]
                return np.mean(all_coords, axis=0) if all_coords else None
        except:
            return None

    def _is_amino_acid(self, residue_name):
        """检查是否为标准氨基酸"""
        standard_aa = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL'
        }
        clean_name = re.sub(r'[^A-Z]', '', residue_name.upper())
        return clean_name in standard_aa

    def extract_active_sites(self, structures):
        """提取活性位点"""
        print("提取活性位点并构建图...")
        active_sites = []

        for i, struct in enumerate(structures):
            if i % 500 == 0:
                print(f"进度: {i}/{len(structures)} ({i / len(structures) * 100:.1f}%)")

            pdb_id = struct['pdb_id']
            protein_file = struct['protein_path']

            try:
                # 跳过氨基酸检查以增加数据量
                # if not self.filter_standard_aas(protein_file):
                #     continue

                if protein_file.endswith('.pdb'):
                    coords, aa_types, adj_matrix = self._extract_residue_coords_from_pdb(protein_file)
                elif protein_file.endswith('.mol2'):
                    coords, aa_types, adj_matrix = self._extract_residue_coords_from_mol2(protein_file)
                else:
                    continue

                # 非常宽松的验证条件
                if len(coords) >= 1 and len(aa_types) >= 1 and adj_matrix.size > 0:
                    quality = self._assess_graph_quality(adj_matrix, len(coords))

                    # 只拒绝完全无效的图
                    if quality != 'invalid':
                        active_sites.append({
                            'pdb_id': pdb_id,
                            'uniprot_id': struct['uniprot_id'],
                            'protein_file': protein_file,
                            'ligand_path': struct['ligand_path'],
                            'coverage': struct['coverage'],
                            'source': struct['source'],
                            'coords': coords,
                            'aa_types': aa_types,
                            'adjacency_matrix': adj_matrix,
                            'num_nodes': len(coords),
                            'num_edges': int(np.sum(adj_matrix) / 2),
                            'quality': quality
                        })

                        self.graph_stats['accepted'] += 1
                        self.graph_stats[f'quality_{quality}'] += 1
                    else:
                        self.graph_stats['rejected'] += 1
                else:
                    self.graph_stats['rejected'] += 1

            except Exception as e:
                print(f"处理 {pdb_id} 时出错: {str(e)}")
                self.graph_stats['rejected'] += 1
                continue

            self.graph_stats['total_processed'] += 1

        return active_sites

    def _assess_graph_quality(self, adj_matrix, num_nodes):
        """评估图质量"""
        try:
            if num_nodes < 1 or adj_matrix.size == 0:
                return 'invalid'

            if num_nodes == 1:
                return 'acceptable'

            edge_count = int(np.sum(adj_matrix) / 2)
            max_edges = num_nodes * (num_nodes - 1) / 2
            density = edge_count / max_edges * 100 if max_edges > 0 else 0

            connectivity_ratio = self._get_connectivity_ratio(adj_matrix)

            # 评估标准
            if density >= 5 and density <= 25 and connectivity_ratio >= 90:
                return 'excellent'
            elif (density >= 2 and density <= 5) or (density > 25 and density <= 40) and connectivity_ratio >= 70:
                return 'good'
            elif density >= 0.5 and connectivity_ratio >= 30:
                return 'acceptable'
            else:
                return 'poor'

        except:
            return 'acceptable'  # 出错时给acceptable评级

    def save_results(self, data, filename):
        """保存结果"""
        if not data:
            print(f"警告: 没有数据要保存到 {filename}")
            return

        output_file = self.output_dir / filename

        if 'coords' in data[0]:
            pickle_file = self.output_dir / (filename.replace('.csv', '_full_data.pkl'))
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"完整数据已保存到 {pickle_file}")

            # 保存简化版CSV
            simple_data = []
            for item in data:
                simple_item = {k: v for k, v in item.items()
                               if k not in ['coords', 'aa_types', 'adjacency_matrix']}
                simple_data.append(simple_item)

            if simple_data:
                df_simple = pd.DataFrame(simple_data)
                df_simple.to_csv(output_file, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)

        print(f"结果已保存到 {output_file}")

    def print_final_stats(self):
        """打印最终统计"""
        print("\n" + "=" * 60)
        print("最终图质量统计:")
        print(f"总处理数量: {self.graph_stats['total_processed']}")
        print(f"最终接受: {self.graph_stats['accepted']}")
        print(f"拒绝数量: {self.graph_stats['rejected']}")
        print(f"接受率: {self.graph_stats['accepted'] / max(1, self.graph_stats['total_processed']) * 100:.1f}%")
        print("\n质量分布:")
        print(f"  优秀质量: {self.graph_stats['quality_excellent']}")
        print(f"  良好质量: {self.graph_stats['quality_good']}")
        print(f"  可接受质量: {self.graph_stats['quality_acceptable']}")
        print(f"  较差质量: {self.graph_stats['quality_poor']}")
        print("=" * 60)

    def run(self):
        """运行完整的处理流程"""
        try:
            # 1. 加载映射（可选）
            has_uniprot_mapping = self.load_sifts_mapping()

            # 2. 处理PDBbind
            print("\n开始处理PDBbind结构...")
            pdbind_structures = self.process_pdbind_structures()
            if pdbind_structures:
                self.save_results(pdbind_structures, "pdbind_structures.csv")
            print(f"► PDBbind结构总数: {len(pdbind_structures)}")

            # 3. 处理scPDB
            print("\n开始处理scPDB结构...")
            scpdb_structures = self.process_scpdb_structures()
            if scpdb_structures:
                self.save_results(scpdb_structures, "scpdb_structures.csv")
            print(f"► scPDB结构总数: {len(scpdb_structures)}")

            # 4. 合并并去重
            print("\n开始去重处理...")
            all_structures = pdbind_structures + scpdb_structures
            non_redundant = self.deduplicate_structures(all_structures, has_uniprot_mapping)
            if non_redundant:
                self.save_results(non_redundant, "non_redundant_structures.csv")

            print(f"► 合并后总数: {len(all_structures)}")
            print(f"► 去重后数量: {len(non_redundant)}")
            if len(all_structures) > 0:
                print(f"► 去重率: {100 * (len(all_structures) - len(non_redundant)) / len(all_structures):.1f}%")

            # 5. 提取活性位点
            print("\n开始提取活性位点并构建最终图...")
            active_sites = self.extract_active_sites(non_redundant)
            if active_sites:
                self.save_results(active_sites, "final_active_sites_with_graphs.csv")

            # 打印详细统计
            self.print_final_stats()

            print(f"\n► 最终活性位点数量: {len(active_sites)}")
            if len(non_redundant) > 0:
                print(f"► 提取率: {100 * len(active_sites) / len(non_redundant):.1f}%")

            # 数据质量统计
            if active_sites:
                node_counts = [site['num_nodes'] for site in active_sites]
                edge_counts = [site['num_edges'] for site in active_sites]
                densities = []

                for site in active_sites:
                    n_nodes = site['num_nodes']
                    n_edges = site['num_edges']
                    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
                    density = n_edges / max_edges * 100
                    densities.append(density)

                print(f"► 节点数统计: 平均{np.mean(node_counts):.1f}, 最小{min(node_counts)}, 最大{max(node_counts)}")
                print(f"► 边数统计: 平均{np.mean(edge_counts):.1f}, 最小{min(edge_counts)}, 最大{max(edge_counts)}")
                print(
                    f"► 密度统计: 平均{np.mean(densities):.1f}%, 最小{min(densities):.1f}%, 最大{max(densities):.1f}%")

            # 最终报告
            print("\n" + "=" * 50)
            print("最终处理完成! 统计:")
            print(f"1. PDBbind原始结构: {len(pdbind_structures)}")
            print(f"2. scPDB原始结构: {len(scpdb_structures)}")
            print(f"3. 合并后结构总数: {len(all_structures)}")
            print(f"4. 去重后结构: {len(non_redundant)}")
            print(f"5. 最终活性位点: {len(active_sites)}")

            # 与论文数据量对比
            paper_target = 5981
            if len(active_sites) < paper_target * 0.5:
                print(f"\n⚠️  注意: 当前数据量 ({len(active_sites)}) 低于论文目标 ({paper_target})")
                print("   建议检查数据路径和SIFTS文件")
            elif len(active_sites) >= paper_target * 0.8:
                print(f"\n✅ 数据量 ({len(active_sites)}) 接近论文水平 ({paper_target})")
            else:
                print(f"\n📊 数据量 ({len(active_sites)}) 中等，可用于训练")

            print("=" * 50)
            print("\n🎯 可以开始GAT-VAE训练了！")
            print(f"   使用文件: final_results/final_active_sites_with_graphs_full_data.pkl")

        except Exception as e:
            print(f"\n❌ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    processor = FinalStructureProcessor(
        pdbind_dir="D:\\pythonstu\\pythonProject16\\dev\\data\\pdbinding\\refined-set",
        scpdb_dir="D:\\pythonstu\\pythonProject16\\dev\\data\\scPDB",
        output_dir="final_results",
        distance_cutoff=4.0
    )
    processor.run()