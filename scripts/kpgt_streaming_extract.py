import argparse
import csv
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
    invalid = [index for index, graph in enumerate(graphs) if graph is None]
    if invalid:
        raise ValueError(f"KPGT graph construction failed for {len(invalid)} SMILES in the current chunk; first chunk-local index: {invalid[0]}")
    return graphs


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
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    total_rows = count_rows(input_csv)
    print(f"Streaming KPGT features for {total_rows} drugs from {input_csv}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, collator = build_model(args.config, Path(args.model_path), device)
    feature_memmap = None
    feature_npy = output_npz.with_suffix(".fps.tmp.npy")
    write_offset = 0

    with torch.no_grad():
        reader = pd.read_csv(input_csv, usecols=[args.smiles_column], chunksize=args.chunk_size)
        for chunk_index, chunk in enumerate(reader, start=1):
            smiles = chunk[args.smiles_column].astype(str).tolist()
            print(f"KPGT chunk {chunk_index}: building {len(smiles)} drug graphs", flush=True)
            graphs = build_graphs(smiles, args.path_length, args.n_jobs)
            fingerprints = build_fingerprints(smiles)
            descriptors = build_descriptors(smiles, args.n_jobs)
            dataset = ChunkMoleculeDataset(smiles, graphs, fingerprints, descriptors)
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
            feature_memmap[write_offset : write_offset + len(features)] = features
            write_offset += len(features)
            feature_memmap.flush()
            print(f"KPGT progress: {write_offset}/{total_rows} drugs ({(write_offset / total_rows) * 100:.1f}%)", flush=True)

    if feature_memmap is None:
        raise ValueError(f"No drugs found in {input_csv}")
    if write_offset != total_rows:
        raise RuntimeError(f"KPGT wrote {write_offset} feature rows but expected {total_rows}")

    np.savez(output_npz, fps=np.asarray(feature_memmap))
    feature_npy.unlink(missing_ok=True)
    print(f"The extracted features were saved at {output_npz}", flush=True)


if __name__ == "__main__":
    main()
