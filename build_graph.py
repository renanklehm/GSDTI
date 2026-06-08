import argparse

from dataset_config import get_dataset_paths
from pipeline import PreparationConfig, _generate_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate target graph files for a dataset.')
    parser.add_argument('--dataset', default='BindingDB', help='Dataset name under data/')
    parser.add_argument('--artifacts-dir', help='Root directory for canonical datasets and derived artifacts')
    parser.add_argument('--force', action='store_true', help='Regenerate graph files')
    args = parser.parse_args()

    paths = get_dataset_paths(args.dataset, artifacts_dir=args.artifacts_dir)
    _generate_graphs(paths, PreparationConfig(artifacts_dir=args.artifacts_dir, force=args.force))


if __name__ == '__main__':
    main()
