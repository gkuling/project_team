# project_team roadmap

The package is being shaped into a teaching-quality research harness. This
roadmap orders the remaining work by dependency and by
teaching-value-per-effort. Phase 0 records the hardening pass that already
landed so the history is legible.

## Phase 0 — Done (hardening pass, August 2026)

An eight-reviewer code critique plus an adversarial audit produced ~80
fixes, landed with a ~90-test CPU-only pytest suite and GitHub Actions CI.
Highlights, in case old results need revisiting:

| Area | What changed |
|---|---|
| Config save/load | `save_pretrained` rewritten on transformers' public API (the old copy of 4.x internals crashed on 4.30, 4.40 and 5.x). `from_pretrained` now reads `<ClassName>.json`; previously it silently returned an all-defaults config — **config reloading had never worked**, so any "resumed" run restarted from step 0. |
| Checkpoint state | `trained_steps` / `best_vl_loss` / `best_vl_step` / `iteration` round-trip; HP-tuning re-saves its counter, making search resume real. |
| Metrics | **`Spec.` was computing precision, not specificity** — historical `Spec.` numbers are precision values. Per-class metric alignment fixed for non-contiguous label sets. |
| Training | `steplr` scheduler was never attached (silent constant LR); subclass y-transforms were silently dropped outside the `'auto'` path; warmup-as-int crashed; `vl_interval` could be 0. |
| Data layer | Silent whole-dataset loss on preload failures now raises; one-hot with pandas labels works; pad transform reachable and correct (axis/fill/2D-reverse); silo sentinels can no longer swallow genuine data. |
| Onboarding | Examples run from any machine (portable working_dir, MNIST under the run folder); `.gitignore` no longer ignores every saved config; packaging metadata is truthful; `transformers>=4.30,<5` pinned. |

Deprecated aliases kept for old code (remove in Phase 6):
`is_Primitive`, `isititerable`, `get_reciprical`, `AddGaussainNoise`,
`default_arguements` (module), `MNIST_CNN_config.input_shape` (property).

## Phase 1 — Practitioner dedup and training UX (needs the green test suite)

- Collapse the ~7 copies of the best-model save block in
  `PT_Practitioner.train_model` into `_save_final_model_state()` +
  `_maybe_save_checkpoint()` helpers.
- Hoist `validate_model` into `PT_Practitioner` (the two children are
  byte-identical apart from comments).
- Hoist `run_inference` with per-child hooks (`_default_output_style()`,
  `postprocess_output(pred, style)`); note the children diverge more than
  they look — decide per-style behavior first.
- `project_team.log(source, message, level)`: a thin wrapper that prints in
  today's format but is backed by `logging`, so instructors can silence
  verbose preload/training output without editing library code. One
  dedicated pass over every `print('<X> Message: ...')`.
- Persist loss history (`tr_loss`/`vl_loss` per interval) on the config —
  bounded — so students can plot a learning curve after `train_model()`.
- Kendall-tau CI: replace the Pearson Fisher-z SE approximation (see the
  NOTE in `Ordinal_Correlation_Practitioner`) after a literature check;
  changing it changes previously reported CI widths.

## Phase 2 — Device handling and structural cleanups

- `torch.device` + a `_to_device()` helper replacing the ~33 ad-hoc
  `torch.cuda.is_available()` branches; groundwork for MPS (Apple Silicon).
- Template-method `get_dataset` in `_Processor` (a `dataset_class`
  attribute) replacing the twin overrides in Image/Text processors.
- `ROCAnalysis_Practitioner_config` so ROC analysis matches every other
  practitioner's config-driven constructor (no in-repo callers, contained
  change). Also resolve the flagged `'X'`-to-prediction column rename in
  its `evaluate`.
- `regresser` → `regressor` on `PTRegressionModel` **with** a
  `_load_from_state_dict` key-migration shim (a bare rename breaks every
  saved checkpoint; a property alias does not preserve state_dict keys).
- Move the examples into `examples/` with a shared MNIST data-prep helper
  (blocked on relocating `default_arguments.py`, which is not part of the
  installed package).

## Phase 3 — SKLearn_Manager + SKLearn_Practitioner

Demonstrates the framework's generality beyond pytorch:
- `SKLearn_Manager(IO_Manager)` mirrors `Pytorch_Manager`'s surface:
  `model_save_pretrained` via `joblib.dump(estimator, 'final_model.joblib')`
  plus the same three config JSONs; `check_if_model_trained` looks for the
  joblib file. No `IO_Manager` dispatch change needed (dispatch is on the
  io-config type).
- `SKLearn_Practitioner` whose `train_model` assembles X/y matrices from
  the preloaded dataset silo and calls `estimator.fit`.

## Phase 4 — SITK_Processor revival (medical imaging)

- Resurrect from history: `git show bc94b4d^:src/project_team/dt_project/DataProcessors/SITK_Processor.py`
  (324 lines) plus the deleted `*_sitk` transforms.
- Conform to the `_Processor` contract (`get_dataset`, default
  `pre_transforms`) and the dict-in/dict-out `field_oi` convention with
  `*_meta_data` keys and `get_reciprocal` chains for invertibility (this
  also resolves the `Reverse_Resample_Image` stub).
- SimpleITK as an optional extra: `pip install project_team[sitk]`,
  import-guarded.

## Phase 5 — Documentation for students

- Expand [GLOSSARY.md](GLOSSARY.md) (started).
- An annotated line-by-line walkthrough of
  `MNIST_classification_TrainTestSplit.py` explaining why each config is
  built the way it is.
- An architecture diagram covering the `exp_type` delegation and the
  files_silo/catalogue indirection.
- A common-errors page mapping the exceptions a student will actually hit
  (wrong config type, missing column, failed pretransform) to plain-English
  causes.
- A steps-vs-epochs note (`n_steps` / `n_epochs` / `vl_interval` /
  `n_saves` interactions).
- A "what your saved config JSON means" page (relevant now that saves
  write the complete record).

## Phase 6 — Packaging modernization

- `pyproject.toml` (a previous attempt lives in git history at `442d25a`),
  single-sourced version, CHANGELOG.md.
- `ruff` in CI (its rules catch the `x is tuple` class of bug that seeded
  this whole effort).
- Remove the deprecated aliases listed in Phase 0; delete the
  `default_arguements` shim module.
- Optional `workflow_dispatch` CI job that runs one example end-to-end.

## Appendix — Known deferred issues

- `Dataset_Fingerprint.average_dictionaries` computes a running
  average-of-averages, not the true dataset mean (needs a Welford-style
  accumulator).
- `MxMnNormalize_Numpy` produces NaN on an all-zero channel
  (`clipped /= clipped.max()`), and in-place ops can fail on integer
  dtypes.
- `_HyperParameterTuning.record_performance` crashes if called before
  `get_gridpoint_args` (`self.timer` is None).
- `io_hptuning_config` accepts a callable `penultimate`, but a callable
  cannot be saved to json (the sanitizer writes a placeholder); consider a
  registry of named penultimate functions.
- Affine augmentations are 3D-only while the flagship examples are 2D
  (examples work because `affine_aug=False`); a 2D path would let the
  MNIST examples demonstrate augmentation.
- `iteration` → `current_iteration` rename (serialized; needs a migration
  shim, and beware `"iteration"` is a substring of `"iterations"`).
