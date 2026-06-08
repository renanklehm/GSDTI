import argparse

from pipeline import PreparationConfig, TrainingConfig, prepare_dataset, run_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare BindingDB and DAVIS_processed, then train/evaluate.')
    parser.add_argument('--artifacts-dir', help='Root directory for canonical datasets and derived artifacts')
    parser.add_argument('--output-dir', help='Directory where prediction CSVs are saved')
    parser.add_argument('--kpgt-dir', help='Path to an external KPGT checkout')
    parser.add_argument('--kpgt-model-path', help='Path to the pretrained KPGT model file')
    parser.add_argument('--force', action='store_true', help='Regenerate derived artifacts')
    args = parser.parse_args()

    prep_config = PreparationConfig(
        artifacts_dir=args.artifacts_dir,
        kpgt_dir=args.kpgt_dir,
        kpgt_model_path=args.kpgt_model_path,
        force=args.force,
    )
    prepare_dataset('BindingDB', config=prep_config)
    prepare_dataset('DAVIS_processed', config=prep_config)
    run_training(
        'BindingDB',
        test_dataset_name='DAVIS_processed',
        training_config=TrainingConfig(epochs=30),
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        output_name='davis',
    )
