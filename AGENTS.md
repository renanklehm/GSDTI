# AGENTS.md

## Repo Shape
- This repo is a script-driven research prototype, not a packaged library. The main entrypoint is now `main.py`, which dispatches into `pipeline.py`.
- The legacy scripts `train_bd_intracl.py`, `train_davis_intracl.py`, `protfeature.py`, and `build_graph.py` are thin wrappers around the unified pipeline.
- `model.py` implements the paper's core model: KPGT drug embeddings + ESM residue embeddings + GCN + `GraphMultisetTransformer` pooling + bilinear classifier.

## Environment
- Separate the local coding environment from the actual training runtime. OpenCode sessions here run in a local `uv` setup on Python 3.14 for editing only.
- Trust checked-in runtime config over the local coding environment and over the README badge: `.python-version` is `3.11`, and both `pyproject.toml` and `uv.lock` require `>=3.11`.
- For the ML stack, use `environment.yml`; it is the only file that declares the actual training dependencies (`torch`, `torch-geometric`, `rdkit`, `esm`, `tmscoring`, etc.).
- `pyproject.toml` is not enough to run training; it only lists lightweight local tooling (`jupyter`, `markitdown[pdf]`).
- Real model runs are expected to happen on a GPU-enabled Google Colab instance configured from `environment.yml`, not in the local OpenCode `uv` environment.
- The Conda env name in `environment.yml` is `drug2`, so `conda activate drug2` matches the checked-in config.
- The checked-in Colab bootstrap notebook is `GSDTI.ipynb`; it assumes a fresh VM, creates a `drug2` micromamba env for GS-DTI/ESM/OpenFold and a separate `kpgt` env for KPGT+DGL, then runs `main.py` with `--kpgt-python /content/micromamba/envs/kpgt/bin/python`.
- The `kpgt` micromamba env in `GSDTI.ipynb` uses `pytorch=1.13.1`, `pytorch-cuda=11.7`, `dgl-cuda11.7=0.9.1post1`, plus `mkl=2024.0` and `intel-openmp` from `defaults`; this avoids both Colab CPU-only PyTorch solves and the `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` import failure seen with the older KPGT stack.
- The `drug2` micromamba env in `GSDTI.ipynb` needs `omegaconf` in addition to `fair-esm`; `esm.pretrained.esmfold_v1()` imports it at runtime when generating ESMFold structures.
- The `drug2` OpenFold/ESMFold stack also needs `dllogger`; `GSDTI.ipynb` installs it from `git+https://github.com/NVIDIA/dllogger.git` alongside the other `fair-esm` extras.
- The OpenFold install in `GSDTI.ipynb` must match the upstream ESM recommendation: install `openfold` from commit `4b41059694619831a7db195b7e0988fc4ff3a307`, or ESMFold can fail at load time with missing structure-module keys.
- On the current Colab/PyTorch 2.4 stack, that OpenFold commit still needs a tiny notebook-side source patch before install: replace `-std=c++14` with `-std=c++17` in `setup.py`, or the CUDA extension build fails with `C++17 or later compatible compiler is required to use PyTorch`.
- Some Colab/OpenFold combinations on the current notebook stack can still fail at `openfold.model.primitives` with `AttributeError: module 'importlib' has no attribute 'util'`; `GSDTI.ipynb` now applies a tiny source patch to `openfold/model/primitives.py` so `importlib.util` is imported explicitly before install.
- That OpenFold commit also expects an older `pytorch_lightning` import path for `seed_everything`; `GSDTI.ipynb` patches `openfold/utils/seed.py` to import it from `lightning_fabric.utilities.seed` for compatibility with the current Colab env.

## Data And Entry Points
- The checked-in sample dataset is under `data/BindingDB/`, with `drugs/drugs.csv`, `targets/targets.csv`, and `df_less1000.csv`.
- `main.py` supports `prepare`, `train`, and `run` subcommands. Use it for both the checked-in BindingDB sample and custom datasets.
- Custom datasets can be ingested from `.csv` or `.parquet` with user-specified `smiles`, `sequence`, and `activity` columns. The pipeline rewrites them into the repo's canonical `Drug_ID/Drug/Target_ID/Target/Y/Label/Target_Length` layout under `data/<dataset>/`.
- The unified pipeline can also place canonical datasets and derived artifacts under a custom root passed via `--artifacts-dir` instead of the repo-local `data/` directory.
- The unified pipeline creates the expected derived artifacts under `<artifacts-dir>/<dataset>/drugs` and `<artifacts-dir>/<dataset>/targets`: `kpgt_base.npz`, `prot_rep.pkl`, target graph `.pt` files, `drug_simmatrix.npz`, and `target_simmatrix.npz`.
- There are no artifact fallbacks anymore. Missing `kpgt_base.npz`, `prot_rep.pkl`, ESMFold `.pdb` files, graph `.pt` files, or similarity matrices must be generated with the real KPGT / ESM-2 / ESMFold flow or preparation fails.
- True KPGT generation requires an external KPGT checkout and model path passed through the CLI, since this repo does not vendor KPGT itself.
- When the Colab runtime uses a separate KPGT environment, pass `--kpgt-python` so preprocessing and feature extraction run with that interpreter rather than with the main GS-DTI runtime.
- `pipeline.py` now resolves KPGT helper scripts from either the repo root or `KPGT/scripts/`, so Colab setup does not need to copy `preprocess_downstream_dataset.py` or `extract_features.py` into the KPGT root.
- `pipeline.py` also injects the KPGT repo root into `PYTHONPATH` for those subprocesses, because upstream KPGT scripts import modules as `from src...` and fail if launched without the repo root on the Python path.
- `train_bd_intracl.py` now prepares `BindingDB` and then calls the unified trainer with `epochs=1`.
- `train_davis_intracl.py` now expects a dataset folder named `data/DAVIS_processed/` and routes both training and evaluation through the unified trainer.

## Path Gotchas
- Centralize dataset paths through `dataset_config.py`; do not reintroduce hard-coded `/GS-DTI/...` or dataset-specific relative paths in individual scripts.
- `dataset_config.get_dataset_paths()` normalizes dataset directory lookup across case variants such as `BindingDB` versus `bindingdb`, and also accepts a custom artifacts root.
- Results are written under the repo-local `results/` directory by default, or under the CLI `--output-dir` when provided.
- `sim_matrix.py` no longer contains a dataset-specific runnable block; call it through `pipeline.py` / `main.py` or import its helpers.

## Runtime Constraints
- The dataset classes now keep tensors on CPU until the training loop moves them onto `device`, so CPU execution is no longer blocked by `.cuda()` calls inside `__getitem__`.
- Most scripts load large artifacts at import/startup time (`np.load`, `pd.read_csv`, `torch.load`), so missing files fail immediately before any training loop starts.
- `pipeline.py` configures the shared `argparse` help formatter with `width=180` for the top-level parser and its `prepare`/`train`/`run` subcommands, so CLI help text wraps much later than the Python default.
- `prepare` now performs true preprocessing only: ESM-2 residue embeddings are generated in-process, ESMFold `.pdb` files are generated in-process, and KPGT features are generated by shelling out to an external KPGT checkout. This is GPU-heavy and, in the checked-in Colab flow, runs with `drug2` for GS-DTI/ESM/OpenFold and a separate `kpgt` env for KPGT+DGL.
- `prepare` now shows `tqdm` progress bars for the overall stage flow and for per-target ESM-2 embedding generation, ESMFold `.pdb` generation, and protein graph building. Existing resume semantics are unchanged: already-generated artifacts are still skipped unless `--force` is used.
- `pipeline.py` writes `tqdm` bars to `stdout`, forces them enabled by default, and keeps the per-target bars visible after completion so they still show up when `prepare` is launched from a Colab `!python ...` subprocess.
- `prepare` also emits persistent `print()` logs before each stage and when stages are skipped or about to generate artifacts, because Colab subprocess output does not reliably preserve `tqdm` redraws.
- `pipeline.py` applies a small compatibility patch to KPGT's `rdNormalizedDescriptors.py` before preprocessing so SciPy's modern `gibrat` name still satisfies KPGT's legacy `gilbrat` lookup.
- `train_bd_intracl.py` currently trains for `epochs = 1`; `train_davis_intracl.py` uses `epochs = 30`.

## Verification
- There is no checked-in test, lint, or typecheck configuration. Do not invent `pytest`, `ruff`, or `mypy` workflows for this repo.
- The most realistic verification is a focused CLI smoke check such as `python main.py --help` and, in the full Conda runtime from `environment.yml`, `python main.py prepare --dataset BindingDB --kpgt-dir <path> --kpgt-model-path <path>` or `python main.py run --dataset <name> --input <file> --artifacts-dir <path> --output-dir <path> --kpgt-dir <path> --kpgt-model-path <path>`.

## GOLDEN RULE
- If you ever change something, update AGENTS.md accordingly.
