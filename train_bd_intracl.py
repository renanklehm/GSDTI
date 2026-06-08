import argparse

from pipeline import PreparationConfig, TrainingConfig, prepare_dataset, run_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare and train on BindingDB.')
    parser.add_argument('--artifacts-dir', help='Root directory for canonical datasets and derived artifacts')
    parser.add_argument('--output-dir', help='Directory where prediction CSVs are saved')
    parser.add_argument('--kpgt-dir', help='Path to an external KPGT checkout')
    parser.add_argument('--kpgt-model-path', help='Path to the pretrained KPGT model file')
    parser.add_argument('--force', action='store_true', help='Regenerate derived artifacts')
    args = parser.parse_args()

    prepare_dataset(
        'BindingDB',
        config=PreparationConfig(
            artifacts_dir=args.artifacts_dir,
            kpgt_dir=args.kpgt_dir,
            kpgt_model_path=args.kpgt_model_path,
            force=args.force,
        ),
    )
    run_training(
        'BindingDB',
        training_config=TrainingConfig(epochs=1),
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        output_name='bindingdb',
    )
