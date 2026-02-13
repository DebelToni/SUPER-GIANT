= ПРИЛОЖЕНИЕ А - КАТАЛОГ НА АРТЕФАКТИТЕ И РЕСУРСИТЕ

Този раздел събира на едно място основните ресурси, използвани и/или генерирани по време на разработката. Целта е бърза навигация при последваща редакция на дипломния текст и финален печатен вариант.

== A.1 Основни документационни файлове (TiDAR)

- `TiDAR/Docs/TiDAR_docs.md`
- `TiDAR/Docs/Implementation_Notes.md`
- `TiDAR/Docs/Training_experiments.md`
- `TiDAR/Docs/README.md`
- `TiDAR/Docs/JAX_prod.md`

== A.2 Typst отчети в TiDAR/Docs

- `TiDAR/Docs/Bucket_Prefix0_1600_A40_Spot.typ`
- `TiDAR/Docs/Finding_Free_token_slots_A40.typ`
- `TiDAR/Docs/KV_Cache_Policy_A40.typ`
- `TiDAR/Docs/Results_Greedy_runs_135_360.typ`
- `TiDAR/Docs/Results_Sweep_4_runs_smollm135.typ`
- `TiDAR/Docs/TiDAR_105k_Inference_HeadToHead.typ`
- `TiDAR/Docs/TiDAR_AR_A40_H100_HeadToHead.typ`
- `TiDAR/Docs/TiDAR_inference_experiments_summary.typ`
- `TiDAR/Docs/TiDAR_vs_AR_SingleForward_Updated.typ`
- `TiDAR/Docs/TinyStories_TiDAR_losses_Comparison.typ`

== A.3 PDF отчети в TiDAR/Docs

- `TiDAR/Docs/Bucket_Prefix0_1600_A40_Spot.pdf`
- `TiDAR/Docs/Finding_Free_token_slots_A40.pdf`
- `TiDAR/Docs/KV_Cache_Policy_A40.pdf`
- `TiDAR/Docs/Results_Greedy_runs_135_360.pdf`
- `TiDAR/Docs/TiDAR_AR_A40_H100_HeadToHead.pdf`
- `TiDAR/Docs/TiDAR_inference_experiments_summary.pdf`
- `TiDAR/Docs/TiDAR_vs_AR_SingleForward_Updated.pdf`
- `TiDAR/Docs/TinyStories_TiDAR_losses_Comparison.pdf`
- `TiDAR/Docs/TinyStories_TiDAR_Losses_Stable_vs_KL.pdf`

== A.4 Логове от експерименти

- `TiDAR/Docs/Training_logs/` (група текстови логове по експерименти и sweep run-ове)
- `TiDAR/Docs/Benchmark_logs/` (benchmark директории за различни inference политики)

== A.5 Изображения и визуални ресурси

- `docs+archive/images/Docker.png`
- `docs+archive/images/PDF-preview-greedy-results.png`
- `docs+archive/images/MHA_diagram_and_math_formula.png`
- `docs+archive/images/Images_TiDAR_Optimization/Vanilla_speculative_decoding_with_smaller_model.png`
- `docs+archive/images/Images_TiDAR_Optimization/Anchor_TiDAR_KV_cache_per_interation.png`
- `docs+archive/images/Images_TiDAR_Optimization/Anchor_TiDAR_forward_pass.png`
- `docs+archive/images/Images_TiDAR_Optimization/TiDAR_Single_Forward_pass.png`
- `docs+archive/images/Images_TiDAR_Optimization/TiDAR_infernece_mask.png`

== A.6 Ключови кодови entry points

- `GIANT/v2/data_pipeline/build_corpus.py`
- `GIANT/v2/model/Run_training.py`
- `GIANT/v2/model/GiantGPT.py`
- `GIANT/v2/model/Transformer_block.py`
- `GIANT/v2/model/Generate_faster.py`
- `GIANT/v2/model/Chat.py`
- `GIANT/v2/model/jit_inference.py`
- `TiDAR/model/Run_training.py`
- `TiDAR/model/Training_step.py`
- `TiDAR/model/inference.py`
- `TiDAR/model/tidar_core.py`
- `TiDAR/model/tidar_masks.py`
- `TiDAR/model/tidar_utils.py`

== A.7 DevOps и remote execution ресурси

- `CICD/tools/s3.py`
- `CICD/tools/sync-gpu.py`
- `CICD/tools/wait-new-gpu.sh`
- `CICD/Docker/README.md`
- `CICD/Docker/giant-training/Dockerfile`
- `CICD/Docker/giant-training-S3/Dockerfile`

== A.8 Текущ документ (драфт за дипломна работа)

- `docs+archive/Diploma_project_documentation/giant_tidar_diploma/main.typ`

Бележка: Настоящият файл е начална версия (initial draft), предназначена за последващо съкращаване, стилистична редакция и добавяне на финални фигури/диаграми.
