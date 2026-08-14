# project_team glossary

A working vocabulary for reading this codebase. (Stub — to be expanded in
the roadmap's documentation phase.)

- **Manager (`IO_Manager` / `Pytorch_Manager`)** — owns everything on disk:
  where data comes from, where splits, configs, checkpoints, and results
  are saved. Holds one *experiment type* object in `manager.exp_type` and
  forwards unknown attribute lookups to it.
- **Experiment type (`_TrainDeploy`, `_Kfold`, `_HyperParameterTuning`)** —
  the statistical project being run, chosen by which `io_config` subclass
  you construct the manager with.
- **Processor (`Image_Processor`, `Text_Processor`)** — turns rows of a
  dataframe into model-ready examples by applying *pre_transforms*, and
  owns the `tr_dset` / `vl_dset` / `if_dset` datasets.
- **Practitioner (`PT_Practitioner` and friends)** — trains, validates, and
  runs inference; the only object that touches the model.
- **config** — a `project_config` subclass holding every parameter an
  object needs, saved as `<ClassName>.json` in the experiment folder.
  Configs also hold mutable run state (`trained_steps`, `best_vl_loss`),
  which is how checkpoints resume.
- **`ipt`** — the sample dictionary flowing through every transform in
  `dt_processing`; keyed at minimum by `'X'` (often also `'y'`). Transforms
  mutate `ipt[transform.field_oi]` and return the whole dict.
- **`field_oi`** — "field of interest": which key of `ipt` a transform
  operates on (`'X'`, `'y'`, or `'pred_y'` for reverses).
- **pre_transforms vs transforms** — pre_transforms run once per example at
  preload (or on first access); transforms run every `__getitem__` call
  (augmentation, normalization, tensor conversion).
- **files_silo / catalogue / `_silo_fields`** — the preload storage: heavy
  objects live in `files_silo`, rows keep `save_name_<n>` sentinel strings,
  and `_silo_fields` records which fields hold sentinels.
- **dataset fingerprint** — per-channel statistics (mean/std/percentiles)
  accumulated during preload; `'auto'` normalization parameters resolve
  against it.
- **steps, not epochs** — the training loop is driven by `n_steps`
  (iterations); `n_epochs` is translated into steps. `vl_interval` is how
  many steps pass between validation/checkpoint points (`n_saves` of them
  in total).
