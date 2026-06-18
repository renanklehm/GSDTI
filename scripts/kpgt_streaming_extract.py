import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dgllife.utils.io import pmap
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from src.data.collator import Collator_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from src.data.featurizer import N_ATOM_TYPES, N_BOND_TYPES, Vocab, smiles_to_graph_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict
from src.utils import set_random_seed


class ChunkMoleculeDataset(Dataset):
    def __init__(self, smiles, graphs, fingerprints, descriptors):
        self.smiles = smiles
        self.graphs = graphs
        self.fingerprints = torch.from_numpy(fingerprints.astype(np.float32))
        descriptors = descriptors.astype(np.float32)
        self.descriptors = torch.from_numpy(np.where(np.isnan(descriptors), 0, descriptors))
        self.labels = torch.zeros((len(smiles), 1), dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        return self.smiles[index], self.graphs[index], self.fingerprints[index], self.descriptors[index], self.labels[index]


def parse_args():
    parser = argparse.ArgumentParser(description="Stream KPGT feature extraction without materializing a full drug graph cache.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--excluded-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--path-length", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def build_fingerprints(smiles):
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            rows.append(np.zeros(512, dtype=np.float32))
        else:
            rows.append(np.asarray(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)), dtype=np.float32))
    return np.vstack(rows).astype(np.float32)


def build_descriptors(smiles, n_jobs: int):
    generator = RDKit2DNormalized()
    if n_jobs == 1:
        rows = [generator.process(value) for value in smiles]
    else:
        with Pool(n_jobs) as pool:
            rows = list(pool.imap(generator.process, smiles))
    return np.asarray(rows)[:, 1:]


def build_graphs(smiles, path_length: int, n_jobs: int):
    if n_jobs == 1:
        graphs = [smiles_to_graph_tune(value, max_length=path_length, n_virtual_nodes=2) for value in smiles]
    else:
        graphs = pmap(smiles_to_graph_tune, smiles, max_length=path_length, n_virtual_nodes=2, n_jobs=n_jobs)
    return graphs


def write_exclusions(excluded_csv: Path, rows, invalid_indices, reason: str) -> None:
    if not invalid_indices:
        return

    excluded_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not excluded_csv.exists()
    excluded_rows = rows.iloc[invalid_indices].copy()
    excluded_rows.insert(0, "source_row", excluded_rows.index.astype(int))
    excluded_rows["exclusion_reason"] = reason
    excluded_rows.to_csv(excluded_csv, mode="a", header=write_header, index=False)


def initialize_exclusions(input_csv: Path, excluded_csv: Path) -> None:
    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()
    header = ["source_row", *columns, "exclusion_reason"]
    excluded_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=header).to_csv(excluded_csv, index=False)


def build_model(config_name: str, model_path: Path, device):
    config = config_dict[config_name]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = LiGhT(
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"],
        path_length=config["path_length"],
        n_heads=config["n_heads"],
        n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    model.eval()
    return model, Collator_tune(config["path_length"])


def load_resume_state(feature_npy: Path, checkpoint_path: Path, total_rows: int, chunk_size: int):
    if not feature_npy.exists():
        return None, 0

    feature_memmap = np.lib.format.open_memmap(feature_npy, mode="r+")
    if feature_memmap.shape[0] != total_rows:
        raise RuntimeError(
            f"Existing KPGT temp feature file has {feature_memmap.shape[0]} rows, expected {total_rows}. "
            f"Delete {feature_npy} to restart from scratch."
        )

    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        completed_rows = int(checkpoint.get("completed_rows", 0))
    else:
        # Backward-compatible recovery for temp files created before checkpointing
        # existed. Unwritten rows in the .npy memmap are zero-filled.
        completed_rows = 0
        saw_empty_row = False
        scan_batch_size = 8192
        for start in range(0, total_rows, scan_batch_size):
            stop = min(start + scan_batch_size, total_rows)
            nonzero_rows = np.any(feature_memmap[start:stop] != 0, axis=1)
            if not saw_empty_row:
                empty_indices = np.flatnonzero(~nonzero_rows)
                if len(empty_indices) == 0:
                    completed_rows = stop
                else:
                    completed_rows = start + int(empty_indices[0])
                    saw_empty_row = True
                    if np.any(nonzero_rows[empty_indices[0] :]):
                        raise RuntimeError(
                            f"Could not infer a contiguous KPGT resume prefix from {feature_npy}. "
                            "Delete the temp file to restart from scratch."
                        )
            elif np.any(nonzero_rows):
                raise RuntimeError(
                    f"Could not infer a contiguous KPGT resume prefix from {feature_npy}. "
                    "Delete the temp file to restart from scratch."
                )

        if completed_rows and completed_rows < total_rows and completed_rows % chunk_size:
            print(
                f"KPGT inferred {completed_rows} completed rows from the temp file; "
                f"rewinding to the last full chunk boundary.",
                flush=True,
            )

    if completed_rows < 0 or completed_rows > total_rows:
        raise RuntimeError(f"Invalid KPGT checkpoint row count: {completed_rows}")
    completed_rows -= completed_rows % chunk_size
    if completed_rows:
        print(f"Resuming KPGT features from {completed_rows}/{total_rows} completed drugs", flush=True)
    return feature_memmap, completed_rows


def save_resume_state(checkpoint_path: Path, *, input_csv: Path, output_npz: Path, total_rows: int, completed_rows: int, chunk_size: int) -> None:
    tmp_path = checkpoint_path.with_suffix(".json.tmp")
    payload = {
        "input_csv": str(input_csv),
        "output_npz": str(output_npz),
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "chunk_size": chunk_size,
    }
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(checkpoint_path)


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")

    set_random_seed(22, 1)
    input_csv = Path(args.input_csv)
    output_npz = Path(args.output_npz)
    excluded_csv = Path(args.excluded_csv)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    total_rows = count_rows(input_csv)
    print(f"Streaming KPGT features for {total_rows} drugs from {input_csv}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, collator = build_model(args.config, Path(args.model_path), device)
    feature_npy = output_npz.with_suffix(".fps.tmp.npy")
    checkpoint_path = output_npz.with_suffix(".kpgt_resume.json")
    feature_memmap, write_offset = load_resume_state(feature_npy, checkpoint_path, total_rows, args.chunk_size)
    if write_offset == 0:
        excluded_csv.unlink(missing_ok=True)
        initialize_exclusions(input_csv, excluded_csv)
    elif not excluded_csv.exists():
        raise RuntimeError(
            f"Cannot resume KPGT exclusion tracking: {checkpoint_path} reports {write_offset} completed rows, "
            f"but {excluded_csv} is missing. Delete the temp KPGT files to restart from scratch."
        )

    with torch.no_grad():
        reader = pd.read_csv(input_csv, chunksize=args.chunk_size)
        for chunk_index, chunk in enumerate(reader, start=1):
            smiles = chunk[args.smiles_column].astype(str).tolist()
            if write_offset >= chunk.index[-1] + 1:
                print(f"KPGT chunk {chunk_index}: already complete; skipping {len(smiles)} drugs", flush=True)
                continue

            print(f"KPGT chunk {chunk_index}: building {len(smiles)} drug graphs", flush=True)
            graphs = build_graphs(smiles, args.path_length, args.n_jobs)
            invalid_indices = [index for index, graph in enumerate(graphs) if graph is None]
            if invalid_indices:
                write_exclusions(excluded_csv, chunk, invalid_indices, "KPGT graph construction failed")
                print(
                    f"KPGT chunk {chunk_index}: excluding {len(invalid_indices)} drugs whose graphs could not be constructed",
                    flush=True,
                )

            valid_indices = [index for index, graph in enumerate(graphs) if graph is not None]
            valid_smiles = [smiles[index] for index in valid_indices]
            valid_graphs = [graphs[index] for index in valid_indices]
            if not valid_smiles:
                write_offset = chunk.index[-1] + 1
                save_resume_state(
                    checkpoint_path,
                    input_csv=input_csv,
                    output_npz=output_npz,
                    total_rows=total_rows,
                    completed_rows=write_offset,
                    chunk_size=args.chunk_size,
                )
                print(f"KPGT progress: {write_offset}/{total_rows} input drugs ({(write_offset / total_rows) * 100:.1f}%)", flush=True)
                continue

            fingerprints = build_fingerprints(valid_smiles)
            descriptors = build_descriptors(valid_smiles, args.n_jobs)
            dataset = ChunkMoleculeDataset(valid_smiles, valid_graphs, fingerprints, descriptors)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collator)
            chunk_features = []
            for batched_data in loader:
                _, graph, ecfp, md, _ = batched_data
                fps = model.generate_fps(graph.to(device), ecfp.to(device), md.to(device))
                chunk_features.append(fps.detach().cpu().numpy())

            features = np.concatenate(chunk_features, axis=0)
            if feature_memmap is None:
                feature_memmap = np.lib.format.open_memmap(
                    feature_npy,
                    mode="w+",
                    dtype=features.dtype,
                    shape=(total_rows, features.shape[1]),
                )
            valid_absolute_indices = chunk.index.to_numpy()[valid_indices]
            feature_memmap[valid_absolute_indices] = features
            write_offset = chunk.index[-1] + 1
            feature_memmap.flush()
            save_resume_state(
                checkpoint_path,
                input_csv=input_csv,
                output_npz=output_npz,
                total_rows=total_rows,
                completed_rows=write_offset,
                chunk_size=args.chunk_size,
            )
            print(f"KPGT progress: {write_offset}/{total_rows} input drugs ({(write_offset / total_rows) * 100:.1f}%)", flush=True)

    if feature_memmap is None:
        raise ValueError(f"No drugs found in {input_csv}")
    if write_offset != total_rows:
        raise RuntimeError(f"KPGT processed {write_offset} input rows but expected {total_rows}")

    excluded_indices = set()
    if excluded_csv.exists():
        excluded_df = pd.read_csv(excluded_csv, usecols=["source_row"])
        excluded_indices = set(excluded_df["source_row"].astype(int).tolist())
    valid_mask = np.ones(total_rows, dtype=bool)
    if excluded_indices:
        valid_mask[list(excluded_indices)] = False

    np.savez(output_npz, fps=np.asarray(feature_memmap)[valid_mask])
    feature_npy.unlink(missing_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    print(f"The extracted features were saved at {output_npz}", flush=True)
    if excluded_indices:
        print(f"Excluded {len(excluded_indices)} drugs from KPGT output; details saved at {excluded_csv}", flush=True)


if __name__ == "__main__":
    main()
