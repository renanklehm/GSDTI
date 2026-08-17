import json
import numpy as np
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
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


def compute_drug_fingerprints(drug_df, radius=2, nbits=1024):
    """
    Computes the Morgan fingerprint of every drug (one row per drug, shape
    (num_drugs, nbits)) instead of a dense num_drugs x num_drugs similarity
    matrix. The pairwise Tanimoto similarity is computed later, on the fly,
    only for the small number of drugs in a given training batch (see
    utils.custom_collate_fn) -- so this scales to datasets with millions of
    drugs, where a dense matrix would not fit in memory.
    """
    from rdkit.Chem import rdFingerprintGenerator

    smiles_column = 'SMILES' if 'SMILES' in drug_df.columns else 'smiles'
    drug_smiles = drug_df[smiles_column].tolist()

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    fingerprints = np.zeros((len(drug_smiles), nbits), dtype=np.uint8)
    for i, smiles in enumerate(drug_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # KPGT already rejected unparseable SMILES upstream; a zero row
            # keeps this array positionally aligned with drugs.csv instead of
            # silently shifting every later drug by one.
            continue
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, fingerprints[i])

    return fingerprints


def compute_and_save_drug_fingerprints(drug_df, save_path, radius=2, nbits=1024):
    fingerprints = compute_drug_fingerprints(drug_df, radius=radius, nbits=nbits)
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


def _load_target_similarity(path):
    data = np.load(path)
    if "fps" in data.files:
        return data["fps"]
    return data[data.files[0]]


def _iter_pending_tm_score_pairs(pdb_files, start_row):
    """Yield the upper-triangle pairs that a matrix of size start_row is missing.

    Row-major so that ordered imap results complete one appended target at a
    time, which is what makes the per-row checkpoint below meaningful.
    """
    for j in range(start_row, len(pdb_files)):
        for i in range(j):
            yield (pdb_files[i], pdb_files[j], i, j)


def _load_tm_resume_state(temp_path, resume_path, num_targets, start_row):
    if not temp_path.exists() or not resume_path.exists():
        return None, start_row

    try:
        checkpoint = json.loads(resume_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, start_row

    if int(checkpoint.get("num_targets", -1)) != num_targets or int(checkpoint.get("start_row", -1)) != start_row:
        print(
            f"Discarding stale TM-score resume state at {temp_path}: "
            f"it was written for a different target table.",
            flush=True,
        )
        temp_path.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        return None, start_row

    matrix = np.lib.format.open_memmap(temp_path, mode="r+")
    if matrix.shape != (num_targets, num_targets):
        del matrix
        temp_path.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        return None, start_row

    completed_rows = int(checkpoint.get("completed_rows", start_row))
    if not start_row <= completed_rows <= num_targets:
        del matrix
        temp_path.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        return None, start_row

    print(f"Resuming TM-score matrix from row {completed_rows}/{num_targets}", flush=True)
    return matrix, completed_rows


def compute_and_extend_tm_score_matrix(
    target_df,
    pdb_folder,
    save_path,
    start_row=0,
    num_processes=None,
    chunksize=None,
):
    """Grow an existing square TM-score matrix to cover newly appended targets.

    Only the pairs involving a target at index >= start_row are computed; the
    already-saved (start_row x start_row) block is copied verbatim. Progress is
    checkpointed after every appended target into a temp .npy memmap plus a
    resume JSON, so an interrupted run picks up at the last completed row
    instead of restarting an O(n^2) job from zero.
    """
    save_path = Path(save_path)
    id_column = 'Gene' if 'Gene' in target_df.columns else 'Target_ID'
    target_ids = target_df[id_column].astype(str).tolist()
    num_targets = len(target_ids)
    pdb_files = [os.path.join(str(pdb_folder), f"{target_id}.pdb") for target_id in target_ids]

    if start_row < 0 or start_row > num_targets:
        raise ValueError(f"Invalid TM-score start row {start_row} for {num_targets} targets")
    if start_row == num_targets:
        print(f"TM-score similarity matrix already covers all {num_targets} targets", flush=True)
        return

    temp_path = save_path.with_suffix(".tm.tmp.npy")
    resume_path = save_path.with_suffix(".tm_resume.json")
    matrix, completed_rows = _load_tm_resume_state(temp_path, resume_path, num_targets, start_row)

    if matrix is None:
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=(num_targets, num_targets))
        matrix[:] = 0.0
        if start_row:
            if not save_path.exists():
                raise FileNotFoundError(
                    f"Cannot extend a TM-score matrix from row {start_row}: {save_path} does not exist"
                )
            existing = _load_target_similarity(save_path)
            if existing.shape != (start_row, start_row):
                raise ValueError(
                    f"{save_path.name} has shape {existing.shape}, expected ({start_row}, {start_row}) "
                    "to extend it; regenerate the dataset with --force"
                )
            matrix[:start_row, :start_row] = existing
        completed_rows = start_row

    for index in range(num_targets):
        matrix[index, index] = 1.0

    total_tasks = (num_targets * (num_targets - 1) - completed_rows * (completed_rows - 1)) // 2
    pending_rows = max(0, num_targets - completed_rows)

    num_proc = min(num_processes or cpu_count(), max(1, total_tasks))
    resolved_chunksize = _resolve_tm_score_chunksize(total_tasks, num_proc, chunksize)
    print(
        f"Extending TM-score matrix to {num_targets} targets: {pending_rows} new rows, "
        f"{total_tasks} pairs, {num_proc} processes, chunksize {resolved_chunksize}",
        flush=True,
    )

    if total_tasks:
        progress = tqdm(total=total_tasks, desc="TM-score pairs", unit="pair", file=sys.stdout, disable=False)
        with Pool(processes=num_proc) as pool:
            # Ordered imap: results arrive in the row-major order produced by
            # _iter_pending_tm_score_pairs, so consuming exactly `row` results
            # drains one appended target and makes the checkpoint below exact.
            results = pool.imap(
                compute_tm_score,
                _iter_pending_tm_score_pairs(pdb_files, completed_rows),
                chunksize=resolved_chunksize,
            )
            for row in range(completed_rows, num_targets):
                for _ in range(row):
                    i, j, tm_score = next(results)
                    matrix[i, j] = tm_score
                    matrix[j, i] = tm_score
                    progress.update(1)
                matrix.flush()
                _save_tm_resume_state(resume_path, num_targets, start_row, row + 1)
        progress.close()

    np.savez(save_path, np.asarray(matrix))
    del matrix
    temp_path.unlink(missing_ok=True)
    resume_path.unlink(missing_ok=True)
    print(f"TM-score similarity matrix saved to: {save_path}", flush=True)


def _save_tm_resume_state(resume_path, num_targets, start_row, completed_rows):
    payload = {
        "num_targets": num_targets,
        "start_row": start_row,
        "completed_rows": completed_rows,
    }
    tmp_path = resume_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(resume_path)
