import argparse

from dataset_config import get_dataset_paths
from pipeline import PreparationConfig, _generate_protein_features


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate protein residue features for a dataset.')
    parser.add_argument('--dataset', default='BindingDB', help='Dataset name under data/')
    parser.add_argument('--artifacts-dir', help='Root directory for canonical datasets and derived artifacts')
    parser.add_argument('--esm-model-name', default='esm2_t33_650M_UR50D')
    parser.add_argument('--force', action='store_true', help='Regenerate protein features')
    args = parser.parse_args()

    paths = get_dataset_paths(args.dataset, artifacts_dir=args.artifacts_dir)
    _generate_protein_features(
        paths,
        PreparationConfig(
            artifacts_dir=args.artifacts_dir,
            esm_model_name=args.esm_model_name,
            force=args.force,
        ),
    )


if __name__ == '__main__':
    main()
