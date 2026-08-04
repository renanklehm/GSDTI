import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tmscoring
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd



def compute_and_save_tanimoto_matrix(drug_df, save_path):
    """
    计算并保存 Drug 的 Tanimoto 相似度矩阵
    :param drug_df: 包含 'Drug_ID' 和 'Drug' (SMILES) 列的 DataFrame
    :param save_path: 保存相似度矩阵的路径

    NOTE: kept for reference/small datasets, but this materializes a dense
    num_drugs x num_drugs matrix and does not scale past a few tens of
    thousands of drugs (memory is O(N^2)). Use
    compute_and_save_drug_fingerprints for large datasets (e.g. ChEMBL).
    """
    smiles_column = 'SMILES' if 'SMILES' in drug_df.columns else 'smiles'
    drug_smiles = drug_df[smiles_column].tolist()  # 提取 SMILES
    drug_mols = [Chem.MolFromSmiles(smiles) for smiles in drug_smiles]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in drug_mols]

    num_drugs = len(fingerprints)
    tanimoto_matrix = np.zeros((num_drugs, num_drugs))
    for i in range(num_drugs):
        for j in range(num_drugs):
            if i <= j:  # 矩阵对称
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                tanimoto_matrix[i, j] = sim
                tanimoto_matrix[j, i] = sim

    # 保存矩阵并返回
    np.savez(save_path, tanimoto_matrix)
    print(f"Tanimoto similarity matrix saved to: {save_path}")


def compute_and_save_drug_fingerprints(drug_df, save_path, radius=2, nbits=1024):
    """
    Computes and saves the Morgan fingerprint of every drug (one row per
    drug, shape (num_drugs, nbits)) instead of a dense num_drugs x num_drugs
    similarity matrix. The pairwise Tanimoto similarity is computed later,
    on the fly, only for the small number of drugs in a given training
    batch (see utils.custom_collate_fn) -- so this scales to datasets with
    millions of drugs, where a dense matrix would not fit in memory.
    """
    from rdkit.Chem import rdFingerprintGenerator

    smiles_column = 'SMILES' if 'SMILES' in drug_df.columns else 'smiles'
    drug_smiles = drug_df[smiles_column].tolist()
    drug_mols = [Chem.MolFromSmiles(smiles) for smiles in drug_smiles]

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    fingerprints = np.zeros((len(drug_mols), nbits), dtype=np.uint8)
    for i, mol in enumerate(drug_mols):
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, fingerprints[i])

    np.savez(save_path, fps=fingerprints)
    print(f"Drug fingerprints saved to: {save_path}")
    


def compute_tm_score(pair):
    pdb_file_1, pdb_file_2, i, j = pair
    if i == j:
        return (i, j, 1.0)
    try:
        alignment = tmscoring.TMscoring(pdb_file_1, pdb_file_2)
        tm_score = alignment.tmscore(**alignment.get_current_values())
    except Exception as e:
        print(f"Error computing TM-score for {pdb_file_1} and {pdb_file_2}: {e}")
        tm_score = 0.0
    return (i, j, tm_score)


def _iter_tm_score_pairs(pdb_files):
    for i, pdb_file_1 in enumerate(pdb_files):
        for j in range(i + 1, len(pdb_files)):
            yield (pdb_file_1, pdb_files[j], i, j)


def _resolve_tm_score_chunksize(total_tasks, num_proc, chunksize):
    if chunksize is not None:
        return max(1, chunksize)
    if total_tasks == 0:
        return 1
    # TM-score jobs are expensive, but hundreds of thousands of tiny chunks
    # still add scheduler overhead in Colab and similar notebook runtimes.
    return max(16, min(256, total_tasks // max(1, num_proc * 16)))


def compute_and_save_tm_score_matrix_parallel_optimized(target_df, pdb_folder, save_path, num_processes=None, chunksize=None):
    """
    高效并行计算并保存 TM-score 相似度矩阵，支持进度显示与分批写盘
    """
    id_column = 'Gene' if 'Gene' in target_df.columns else 'Target_ID'
    target_ids = target_df[id_column].tolist()
    num_targets = len(target_ids)
    pdb_files = [os.path.join(pdb_folder, f"{target_id}.pdb") for target_id in target_ids]

    # 只生成上三角任务，i==j直接赋1.0，不计算
    total_tasks = num_targets * (num_targets - 1) // 2

    tm_score_matrix = np.zeros((num_targets, num_targets), dtype=np.float32)
    for idx in range(num_targets):
        tm_score_matrix[idx, idx] = 1.0

    num_proc = min(num_processes or cpu_count(), max(1, total_tasks))
    resolved_chunksize = _resolve_tm_score_chunksize(total_tasks, num_proc, chunksize)
    print(f"Using {num_proc} processes, chunksize {resolved_chunksize}, total tasks: {total_tasks}")
    with Pool(processes=num_proc) as pool:
        results = pool.imap_unordered(compute_tm_score, _iter_tm_score_pairs(pdb_files), chunksize=resolved_chunksize)
        for i, j, tm_score in tqdm(results, total=total_tasks):
            tm_score_matrix[i, j] = tm_score
            tm_score_matrix[j, i] = tm_score  # 对称

    np.savez(save_path, tm_score_matrix)
    print(f"TM-score similarity matrix saved to: {save_path}")
