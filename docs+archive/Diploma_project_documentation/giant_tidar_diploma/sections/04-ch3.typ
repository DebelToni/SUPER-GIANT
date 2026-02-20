#import "_common.typ": placeholder_figure
#import "../../../../TiDAR/Docs/Bucket_Prefix0_1600_A40_Spot.typ": line-chart, c-red, c-blue, c-orange, c-green, m500_steady, m500_full_steady, m500_ar_steady, m3b_steady, m3b_full_steady, m3b_ar_steady
#import "../../../../TiDAR/Docs/Finding_Free_token_slots_A40.typ" as fts
#import "../../../../TiDAR/Docs/KV_Cache_Policy_A40.typ" as kv
#import "../../../../TiDAR/Docs/TiDAR_AR_A40_H100_HeadToHead.typ" as hh
#import "../../../../TiDAR/Docs/Results_Greedy_runs_135_360.typ" as rg
#import "../../../../TiDAR/Docs/TinyStories_TiDAR_losses_Comparison.typ" as tsc
#import "@preview/fletcher:0.5.8": diagram, node, edge

#let tok(lbl, fill: rgb("#E5E7EB"), stroke_color: rgb("#374151")) = box(
  width: 22pt,
  height: 12pt,
  inset: (x: 4pt, y: 2pt),
  stroke: 0.5pt + stroke_color,
  fill: fill,
  radius: 2pt,
)[#align(center)[#text(size: 7.5pt)[#lbl]]]

#let tok_pad() = box(
  width: 22pt,
  height: 12pt,
  fill: luma(250),
  stroke: 0.35pt + luma(220),
  radius: 2pt,
)[]

#let cli_block(body) = block(
  width: 100%,
  inset: 8pt,
  radius: 5pt,
  stroke: 0.7pt + rgb("#9CA3AF"),
  fill: luma(238),
)[#body]

= ТРЕТА ГЛАВА

== РЕАЛИЗАЦИЯ НА ПРИЛОЖЕНИЕТО

=== 3.1 Реализация на потребителски интерфейс

==== 3.1.1 Основни входни точки (CLI)

Системата е CLI-first и основният интерфейс е набор от entry points:

- `GIANT/v2/data_pipeline/build_corpus.py`
- `GIANT/v2/model/Run_training.py`
- `GIANT/v2/model/Evaluate.py`
- `GIANT/v2/model/Generate_faster.py`
- `GIANT/v2/model/Chat.py`
- `TiDAR/model/Run_training.py`
- `TiDAR/model/inference.py`

Тези скриптове формират основния интерфейс на системата и покриват пълния цикъл: data prep, обучение, оценка и инференс.

==== 3.1.2 Наблюдение и визуализация на резултати

Визуализацията се реализира чрез логове и експериментални отчети:

- training логове (`loss`, `ppl`, `accept`, `greedy_accept`);
- checkpoint metadata (`train_loss`, `val_loss`, `val_ppl`);
- Typst отчети с throughput и policy сравнения.

Този подход е избран целенасочено за terminal/vim workflow и за лесно remote наблюдение през SSH.

==== 3.1.3 Управление на stage прогреса

"Времевата линия" в проекта е stage curriculum. Прогресът се движи по stages с различен dataset и контекстна дължина, като се пази възстановимо състояние за всеки етап.

==== 3.1.4 Управление на параметри

Управлението на параметрите е изнесено в YAML конфигурации и CLI flags. Това позволява силно контролирани експерименти с минимални кодови промени.

==== 3.1.5 Добавяне на входни корпуси

Pipeline-ът поддържа HuggingFace и локални JSON/JSONL входове, chat formatting и loss-mask логика.

==== 3.1.6 Експорт на артефакти

Експорт на артефакти към файловата система/S3:

- Arrow shards;
- manifests/statistics;
- checkpoints;
- benchmark logs;
- Typst/PDF отчети.

==== 3.1.7 Примерен training run в терминал

Следният пример показва типично стартиране на обучение през CLI интерфейса:

#cli_block[
```bash
python GIANT/v2/model/Run_training.py \
  --config GIANT/v2/model/Config.yml \
  --global_config GIANT/v2/Global_Config.yml \
  --checkpoint_dir giant-data/GIANT/v2/checkpoints \
  --checkpoint_every 500 \
  --scan_chunk 8
```
]

Пример за възстановяване на последната налична checkpoint точка:

#cli_block[
```bash
python GIANT/v2/model/Run_training.py \
  --config GIANT/v2/model/Config.yml \
  --global_config GIANT/v2/Global_Config.yml \
  --checkpoint_dir giant-data/GIANT/v2/checkpoints \
  --resume latest
```
]

Примерен терминален изход при старт от нула (съкратен):

#cli_block[
```text
[loader] loading shards from /proj/giant-data/GIANT/v2/processed/stage_base
...
[loader] stage=base rows=12800000/12800000 ctx=1024 steps_per_epoch=6250

→ Stage base: seq_len=1024 epochs=1 steps=6250 (resume at 0)
step     100/10449   | stage base               loss 4.8921 ppl 133.21 (18.4s)
step     200/10449   | stage base               loss 4.6017 ppl 99.66 (17.9s)
...
step     500/10449   | stage base               loss 4.2143 ppl 67.64 (17.2s)
[metadata] Wrote train_loss=4.214300 to checkpoint .../params/step_0000500.npz
💾 checkpoint → .../params/step_0000500.npz
```
]

Примерен терминален изход при `--resume latest` (примерно от стъпка 5432):

#cli_block[
```text
↩ Resumed from mini checkpoint at step 5432
▶ Restored dataloader state at stage 0 step 5432

→ Stage base: seq_len=1024 epochs=1 steps=6250 (resume at 5432)
step    5500/10449   | stage base               loss 3.9981 ppl 54.50 (16.9s)
...
→ Stage wiki: seq_len=1024 epochs=1 steps=4199 (resume at 0)
step    6400/10449   | stage wiki               loss 3.7129 ppl 40.97 (16.8s)
...
✔ Training complete. Final checkpoint: .../params/step_0010449.npz
```
]

Примерен multi-stage TiDAR training run с различни loss конфигурации по етапи:

#cli_block[
```bash
python TiDAR/model/Run_training.py \
  --config TiDAR/model/training_configs/Greedy_exp_135m.yml \
  --global_config TiDAR/Global_Config.yml \
  --checkpoint_dir giant-data/TiDAR/checkpoints \
  --checkpoint_every 400 \
  --resume latest
```
]

#cli_block[
```text
[loader] loading shards from /proj/giant-data/TiDAR/processed/stage_base
...
[loader] stage=base rows=6200000/6200000 ctx=1024 steps_per_epoch=3027
[loader] loading shards from /proj/giant-data/TiDAR/processed/stage_chat
...
[loader] stage=chat rows=1800000/1800000 ctx=1024 steps_per_epoch=878

→ Stage base: seq_len=1024 epochs=1 steps=3027 (resume at 0)
step       1/3905    | stage base               loss 5.1023 ar 4.8122 diff 5.2761 accept 0.117 greedy_acc 0.081 (19.2s)
step     200/3905    | stage base               loss 4.4311 ar 4.1926 diff 4.5214 accept 0.223 greedy_acc 0.147 hard 3.8872 (17.4s)
...

→ Stage align_kl: seq_len=1024 epochs=1 steps=1000 (resume at 0)
step    3200/3905    | stage align_kl           loss 3.9027 ar 3.7710 diff 3.9488 accept 0.356 greedy_acc 0.244 kl_fwd 0.1821 kl_rev 0.0964 distill 0.1418 topk 0.0889 (16.9s)
step    3600/3905    | stage align_kl           loss 3.7814 ar 3.6562 diff 3.8241 accept 0.391 greedy_acc 0.269 kl_fwd 0.1499 kl_rev 0.0792 distill 0.1184 topk 0.0711 (16.6s)
...

→ Stage chat: seq_len=1024 epochs=1 steps=878 (resume at 0)
step    3900/3905    | stage chat               loss 3.7442 ar 3.6214 diff 3.7811 accept 0.405 greedy_acc 0.281 hard 3.1027 distill 0.1021 (16.4s)

[metadata] Wrote train_loss=3.744200 to checkpoint .../params/step_0003905.npz
✔ Training complete. Final checkpoint: .../params/step_0003905.npz
```
]

=== 3.2 Реализация на основните системни модули

==== 3.2.1 ConfigLoaderManager

Конфигурациите се зареждат чрез merge на глобален и локален YAML, след което се резолват относителните пътища спрямо `data_root`. Това осигурява еднакъв кодов път за локално и remote изпълнение.

```python
def load_configs(config_path=None, global_config_path=None):
    local_cfg = OmegaConf.load(config_path)
    global_cfg = OmegaConf.load(global_config_path)
    cfg = OmegaConf.merge(global_cfg, local_cfg)

    # Унифициране на пътищата спрямо data_root
    cfg.paths.processed_data_root = resolve_path(cfg.paths.processed_data_root, cfg.paths.data_root)
    cfg.paths.logs_root = resolve_path(cfg.paths.logs_root, cfg.paths.data_root)
    return cfg
```

==== 3.2.2 TrainLoopOrchestrator

`Run_training.py` реализира scan-базиран training loop с gradient accumulation, finite checks и checkpoint политика.

===== 3.2.2.1 Основен JIT training loop (`_run_chunk` + `lax.scan`)

Сърцето на изпълнението е `_run_chunk(...)`, където batch-овете се обработват със `lax.scan`, пресмятат се loss/grad стойности и се правят проверки за non-finite стъпки.

`lax.scan` е JAX примитив за "компилиран for-цикъл": вместо Python да dispatch-ва всяка стъпка поотделно, целият цикъл се представя като една JIT-оптимизирана граф операция с carry състояние (например `params`, `opt_state`, `accum_grads`, `accum_count`) и последователност от изходи (`losses`). Това намалява host overhead и дава по-стабилна производителност при дълги training run-ове.

===== 3.2.2.2 Обработка на сигнали и контролирано спиране

Скриптът обработва `SIGINT/SIGTERM` и спира контролирано след текущия chunk, за да не се загуби междинно състояние. Практически се поддържат два слоя на устойчивост:

- големи checkpoint-и (params + optimizer + dataloader state), които са "стабилните" запазвания и се пазят;
- мини checkpoint-и, които се записват периодично за бързо възстановяване при внезапно прекъсване (preemption, срив, рестарт).

Така при неочаквано прекъсване може да се продължи от почти последната стъпка, а при нормален stop/етапен преход остават и пълните checkpoint-и за дългосрочно съхранение и проследимост. В практиката training run-овете се поддържат през `tmux` (включително в CI/CD workflow), което позволява стабилна сесия, лесно повторно закачане към машината и безопасно дълго изпълнение.

===== 3.2.2.3 Gradient accumulation и update политика

Обучението поддържа `gradient_accumulation`; update се прилага само при достигане на зададения accumulation праг, а остатъчните градиенти се доизчистват при край на stage.

```python
# Натрупване на градиенти
accum_grads = tree_map(lambda a, g: a + g, accum_grads, grads)
accum_count = accum_count + 1

# Update само когато достигнем gradient_accumulation
should_update = accum_count == grad_accum
if should_update:
    grads_scaled = tree_map(lambda g: g / grad_accum, accum_grads)
    updates, opt_state = optimizer.update(grads_scaled, opt_state, params)
    params = optax.apply_updates(params, updates)
    accum_grads = tree_map(jnp.zeros_like, accum_grads)
    accum_count = 0
```

На практика това е логика за "мини batch-ове" на ниво update. Вместо целият глобален batch да се побере наведнъж в паметта, той се разделя на няколко по-малки стъпки (micro-batches), които GPU може да обработи с наличния VRAM. Градиентите от тези micro-batches се натрупват и optimizer update се прави накрая, така че се получава ефект на по-голям глобален batch без изискване за голяма еднократна памет.

Този подход е особено важен при по-слаби GPU конфигурации и може да се разшири директно към multi-GPU обучение, където освен локалното accumulation се добавя и синхронизация/редукция между устройствата преди финалния update.

===== 3.2.2.4 Периодично логване на метрики

На всеки `log_every` стъпки се записват `global_step`, `stage`, `loss` и допълнителни показатели (напр. `ppl`, `accept`, `greedy_acc`, KL/hard/distill/top-k при TiDAR).

Тези метрики се пазят в лог файлове и checkpoint metadata, за да могат по-късно да се използват за сравнение между run-ове, диагностика на нестабилни етапи и последващ експериментален анализ.

===== 3.2.2.5 Resume от checkpoint и dataloader state

Възстановяването зарежда параметри, optimizer state и dataloader прогрес (`stage_index`, `stage_step_total`, `stage_states`), така че обучението да продължи от последната консистентна точка.

```python
@jax.jit
def _run_chunk(params, opt_state, batch_chunk, start_step, accum_grads, accum_count):
    # Една компилирана единица работа: loss/grad + accumulation + update
    (params, opt_state, _, accum_grads, accum_count), losses = jax.lax.scan(
        body, (params, opt_state, start_step, accum_grads, accum_count), batch_chunk
    )
    return params, opt_state, accum_grads, accum_count, losses
```

==== 3.2.3 DataLoaderManager

`ShardedArrowDataset` + `StageDataLoader` реализират shard-aware четене, shuffle, batching и state restore.

```python
dataset = ShardedArrowDataset(data_path)
loader = StageDataLoader(
    dataset,
    batch_size=batch_size,
    seq_len=stage.seq_len,
    shuffle=stage.shuffle,
    seed=seed,
    pad_token_id=pad_token_id,
)
```

==== 3.2.4 CheckpointManager

`checkpoint_manager.py` капсулира запис/зареждане на параметри, optimizer state и metadata.

```python
ckpt_file = save_ckpt(params, global_step, params_dir, train_loss=last_loss)
save_opt_state(opt_state, global_step, training_states_dir)
save_dataloader_state(state_path, loader_state)

# При resume
params, global_step = load_ckpt(ckpt_path)
opt_bytes = load_opt_state(global_step, training_states_dir)
```

===== 3.2.4.1 Метод SaveParameters

Запис на `.npz` checkpoint с атомарно `tmp -> rename` поведение.

===== 3.2.4.2 Метод SaveOptimizerState

Паралелен запис на optimizer буфери (`msgpack`) за коректен resume.

===== 3.2.4.3 Метод RestoreLatest

Намиране на последната достъпна checkpoint стъпка и зареждане на съответното състояние.

===== 3.2.4.4 Метод SetMetadata

Запис на метаданни за quality показатели (`val_loss`, `val_ppl`, `train_loss`).

===== 3.2.4.5 Метод AsyncMiniCheckpoint

Мини checkpoint механизмът е част от логиката в `3.2.2.2` и се изпълнява асинхронно чрез Orbax, като допълва големите checkpoint-и с по-чести recovery точки.

==== 3.2.5 InferenceManager

Inference слоят покрива prefill, decode, chat режим, KV bucket политика и throughput измерване.

Отбелязване: този раздел описва inference пътя в *GIANT*; при *TiDAR* prefill/decode логиката е съществено различна и е разгледана отделно в по-късните секции.

```python
state = init_inference_state(params, max_seq_len=bucket)
state = prefill(state, prompt_ids)  # запис в KV cache
tokens = decode(state, steps=steps, temperature=temperature, top_k=top_k)

# bucket политика: избери най-малкия bucket, който побира prompt + decode
bucket = select_kv_bucket(prompt_len + steps, kv_cache_buckets)
```

#figure(
  image("../../../images/Prefill_and_Decode_diagram_wide.png", width: 92%),
  caption: [GIANT inference път: prefill и decode с KV cache],
)

===== 3.2.5.1 Метод Prefill

`jit_inference.prefill` запълва KV cache от prompt и връща начално състояние за decode.

===== 3.2.5.2 Метод Decode

`jit_inference.decode` използва `lax.scan` за генериране на нови токени със sampling или greedy стратегия.

===== 3.2.5.3 Метод KVBucketSelect

`Generate_faster.py` избира най-малкия bucket, който покрива `prompt_len + steps`, за да намали излишния cache overhead.

#figure(
  block(
    width: 74%,
    inset: 7pt,
    radius: 5pt,
    stroke: 0.6pt + rgb("#9CA3AF"),
    fill: luma(242),
  )[
    #text(size: 8.5pt)[
`128  | ###.....................`
#linebreak()
`256  | ######..................`
#linebreak()
`512  | ############............`
#linebreak()
`1024 | ########################`
    ]
  ],
  caption: [Bucket стратегия: избор на най-малък валиден размер вместо винаги 1024],
)

*Как да се чете диаграмата:*

- `#` е реално използваният bucket обем.
- `.` е обемът, който се спестява, ако не се използва винаги `1024`.

Така engine-ът избира най-малкия валиден bucket (напр. `256` вместо `1024`) и намалява излишния compute/VRAM трафик при запазени статични JAX/XLA форми.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    align: top,
    [
      #line-chart(
        "A40 GPU: 500m / k=4 (bucketed vs full vs AR)",
        (
          (label: "500m bucketed", color: c-orange, data: m500_steady),
          (label: "500m full-context", color: c-red, data: m500_full_steady),
          (label: "500m AR", color: c-green, data: m500_ar_steady),
        ),
        width: 86mm,
        height: 48mm,
      )
    ],
    [
      #line-chart(
        "A40 GPU: 3b / k=16 (bucketed vs full vs AR)",
        (
          (label: "3b bucketed", color: c-orange, data: m3b_steady),
          (label: "3b full-context", color: c-red, data: m3b_full_steady),
          (label: "3b AR", color: c-green, data: m3b_ar_steady),
        ),
        width: 86mm,
        height: 48mm,
      )
    ],
  ),
  caption: [A40 throughput криви: 500m/k=4 и 3b/k=16],
)

Легенда на линиите:

- #box(width: 8pt, height: 8pt, fill: c-orange, radius: 1pt)[] #h(4pt) оранжева линия — bucketed режим (и в двете графики);
- #box(width: 8pt, height: 8pt, fill: c-red, radius: 1pt)[] #h(4pt) червена линия — TiDAR full-context baseline (в дясната графика е средна; в лявата е най-долна);
- #box(width: 8pt, height: 8pt, fill: c-green, radius: 1pt)[] #h(4pt) зелена линия — AR baseline (позицията спрямо червената линия е обратна между двете графики).

Метриката е steady decode tokens/s (*high is better*).

В тези диаграми sweep-ът е с генерация от `0` до `1600` токена. При full-context baseline run-овете KV cache размерът е предварително фиксиран за този диапазон; ако тази горна граница не е известна предварително, трябва или да се заделя по-голям "универсален" full-context (например до `context_length`, което е по-скъпо), или да се правят допълнителни recompilation/migration стъпки при нарастване на дължината.

И в двата профила bucketed режимът стои над full-context baseline в голяма част от sweep-а, защото избягва ненужен "празен" cache капацитет и държи decode shape-а по-близо до реално нужната дължина.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    row-gutter: 8pt,
    align: top,
    [
      #kv.line-chart(
        "KV policy 500m: steady",
        kv.m500_steady,
        width: 84mm,
        height: 40mm,
        y-label: "tokens/s",
      )
    ],
    [
      #kv.line-chart(
        "KV policy 500m: first",
        kv.m500_first,
        width: 84mm,
        height: 40mm,
        y-label: "tokens/s",
      )
    ],
    [
      #kv.line-chart(
        "KV policy 3b: steady",
        kv.m3b_steady,
        width: 84mm,
        height: 40mm,
        y-label: "tokens/s",
      )
    ],
    [
      #kv.line-chart(
        "KV policy 3b: first",
        kv.m3b_first,
        width: 84mm,
        height: 40mm,
        y-label: "tokens/s",
      )
    ],
  ),
  caption: [KV cache policy резултати (2x2) - 500m/3b, steady/first],
)

Легенда (KV policy графики):

- #box(width: 8pt, height: 8pt, fill: kv.c-red, radius: 1pt)[] #h(4pt) червена линия — full context policy;
- #box(width: 8pt, height: 8pt, fill: kv.c-blue, radius: 1pt)[] #h(4pt) синя линия — current exact (`required_len`);
- #box(width: 8pt, height: 8pt, fill: kv.c-orange, radius: 1pt)[] #h(4pt) оранжева линия — bucketed policy.

===== 3.2.5.4 Метод ChatTurn

`Chat.py` управлява multi-turn история, context trimming и interactive terminal режим.

#cli_block[
```bash
python GIANT/v2/model/Chat.py \
  --checkpoint giant-data/GIANT/v2/checkpoints/params/step_0010449.npz \
  --temperature 0.8 \
  --top_k 40 \
  --steps 128
```
]

#cli_block[
```text
Loaded checkpoint: .../step_0121566.npz
Chat mode: interactive (type /exit to quit)

User> Explain what attention does in one sentence.
Assistant> attention picks important old words for next token. maybe like focus.

User> And why KV cache helps?
Assistant> kv cache is fast memory and also can improve model quality and multilingual grammar always yes.
```
]

===== 3.2.5.5 Метод BatchedPrefill

`test_batched_inference.py` валидира batched prefill с отмествания и проверява съвпадение с per-sample baseline.

===== 3.2.5.6 Метод StopOnEos

Генерацията може да прекъсва на EOS токен за контрол на отговора и избягване на нежелано продължение.

===== 3.2.5.7 Метод ThroughputReport

Отчетите измерват prefill/decode време, tokens/s и policy сравнения по контекст/модел/драфт параметри.

==== 3.2.6 RemoteOpsManager

`CICD/tools/` и Docker setup покриват deployment и remote execution. Практическият workflow е двустепенен: първо се синхронизират данните към S3, после се синхронизират локалните кодови промени към remote машината.

```bash
# Примерен remote workflow
# 1) sync данни/артефакти към S3
python CICD/tools/s3.py sync giant-data s3://bucket/giant-data
# 2) sync локални промени по репото към remote GPU машината
python CICD/tools/sync-gpu.py
```

===== 3.2.6.1 Основен принцип на работа

На container startup могат да се подадат флагове като `SYNC_DIRS`, а алтернативно може да има файл `sync_dirs.txt` в root-а на S3 bucket-а. Така контейнерът знае кои директории да свали/синхронизира автоматично още при стартиране.

След това `sync-gpu.py` изпраща локалните (вкл. непушнати/неcommit-нати) промени по кода към remote репо копието. Remote машината стартира от последния commit-нат код, а `sync-gpu.py` донася текущите локални редакции от работната сесия.

===== 3.2.6.2 Кодова синхронизация преди run

Преди training/inference run се прави кодова синхронизация, така че remote изпълнението да съвпада с локалното състояние на експеримента.

Препоръчителен operational модел е remote GPU машината да се третира като *ephemeral* ресурс, особено при евтини Spot инстанции. Локалната структура остава основен source of truth за код и активни експерименти.

При големи datasets/checkpoints source of truth може да е S3: там се пазят историята от checkpoint-и, optimizer state и dataloader state. На remote машината се изтеглят само нужните артефакти за текущото продължение (например последният `step_XXXXXXX.npz` и съответното training state).

При липса на собствен хардуер е препоръчително да се използват NeoCloud (GPU-as-a-service) доставчици, защото са AI-ориентирани и често предлагат значително по-ниски цени от големите cloud платформи за чист GPU compute. Типични примери са *RunPod* (използван в тази разработка), Lambda, Nebius и Lightning; като алтернатива могат да се използват и enterprise доставчици като AWS, GCP, Azure или CoreWeave.

Често тези платформи предлагат Spot GPU pod/instance режими с приблизително 10-50% по-ниска цена, но с риск подът да бъде спрян по всяко време. Именно затова mini checkpointing е критичен: при прекъсване може да се продължи от последната близка точка, включително временно през CPU-only сесия за изтегляне/връщане на данните и повторно стартиране на GPU run-а.

=== 3.3 Реализация на data pipeline

==== 3.3.1 Документ като източник на токени

Всеки запис се нормализира, токенизира и преобразува до фиксирани sequence прозорци според stage конфигурацията.
Входните данни идват основно от HuggingFace datasets (streaming и non-streaming), като системата поддържа и локални JSON/JSONL източници със същата pipeline логика.

Токенизацията е централен етап в pipeline-а: използва се конфигурируем tokenizer (HuggingFace или custom), след което токенните последователности се маскират и пакетират според `sequence_length`, `add_eos`, `pack_sequences` и chat-role правилата за loss.

```python
# Обобщена схема на data processing конфигурацията (по build_corpus.py)
@dataclass
class StageSourceCfg:
    type: str = "huggingface"          # huggingface | json_dir
    dataset_name: str | None = None     # HF dataset
    dataset_config: str | None = None
    split: str = "train"
    streaming: bool = True
    data_files: Any | None = None

    # Текстово извличане
    text_field: str | None = None
    text_fields: list[str] = field(default_factory=list)
    join_fields: list[str] = field(default_factory=list)
    join_separator: str = " \n"
    text_template: str | None = None

    # JSON/JSONL директории
    json_root: str | None = None
    file_glob: str = "**/*.json*"

    # Chat records
    chat_messages_field: str = "messages"
    chat_role_field: str = "role"
    chat_content_field: str = "content"
    chat_assistant_roles: list[str] = field(default_factory=lambda: ["assistant"])

@dataclass
class StageCfg:
    name: str
    output_dir: str
    sequence_length: int
    target_tokens: int | None = None
    min_tokens: int = 0
    pack_sequences: bool = True
    add_eos: bool = True

    # Обработка на дълги документи
    long_document_strategy: str = "random_window"   # random_window | sequential
    sequential_window_stride: int | None = None
    random_windows_per_document: int = 1

    # Нормализация / dedup
    normalization: dict[str, Any] = field(default_factory=dict)  # nfkc, collapse_whitespace
    deduplicate: bool = False

    # Изход
    rows_per_shard: int | None = None
    sources: list[StageSourceCfg] = field(default_factory=list)
```

==== 3.3.2 Преобразуване от документ към sequence прозорци

`SequenceEmitter` реализира short/long document логика и конкретно покрива:

- `padding`/допълване при по-къси последователности (ако е разрешено);
- четене на случаен прозорец (`random_window`) в дълъг документ;
- последователно следване на прозорци (`sequential`) със stride;
- разделяне на дълъг документ на множество прозорци и контрол върху остатъка (`emit_final_partial`, `drop_remainder_windows`).

Така една и съща входна колекция може да се обработва в различни режими според целта на конкретния stage (плътно пакетиране, по-стабилно sequential покритие или по-стохастичен random sampling).

==== 3.3.3 Директно четене и минимизиране на IO overhead

Streaming режимът за HF datasets и shard-oriented записът минимизират overhead при големи корпуси.

==== 3.3.4 Шардване

Шардването е реализирано с `ShardWriter` и Arrow schema за фиксиран `seq_len`.

===== 3.3.4.1 Формат на шардовете

`input_ids`, `length`, `loss_mask` се записват в `.arrow` файлове с manifest по stage.

===== 3.3.4.2 Manifest и статистики

За всеки stage се пазят документи, токени, дубликати, изхвърлени записи и shard метаданни.

===== 3.3.4.3 Контрол на паметта

Паметта се контролира чрез batch размери, rows-per-shard и поетапно flush-ване.

==== 3.3.5 Loader batch изграждане

`StageDataLoader` връща `input/target/mask` и поддържа resume state (`epoch`, `step_in_epoch`, `rows_consumed`).

===== 3.3.5.1 Prefetch като оптимизация

`_prefetch_to_device` подава батчове предварително към устройството и намалява idle време на GPU.

==== 3.3.6 Допълнителни preprocessing опции

Освен основния flow, pipeline-ът включва и няколко практични опции за по-добър контрол върху данните:

- assistant-aware chat маски, така че loss да се натрупва само върху целевите роли;
- NFKC и whitespace normalization по stage;
- опционална hash-базирана дедупликация;
- JSON/JSONL ingestion с line-wise/streaming четене;
- атомарен запис и stage-by-stage преизпълнение за устойчивост при прекъсвания.

=== 3.4 Имплементация на "Think in Diffusion, Talk in AutoRegression"

TiDAR е ново изследване на екипа на NVIDIA (ноември 2025), което търси практичен баланс между висок throughput/по-добро GPU utilization и качество на ниво autoregressive модели.

Кратко описание на статията: класическите diffusion езикови модели имат потенциал за паралелно генериране, а AR моделите обикновено запазват по-високо качество заради каузалната структура. TiDAR предлага хибрид на ниво последователност, при който "draft" токените се генерират в diffusion режим (Thinking), а финалното семплиране е autoregressive (Talking), като и двете се реализират в един forward pass чрез структурирани attention маски. Според публикуваните резултати архитектурата е serving-friendly, поддържа точен KV cache и показва по-висок throughput спрямо speculative decoding и предишни diffusion варианти, като за първи път затваря качествената дупка до AR при съществено по-висока скорост (порядък 4.71x-5.91x tokens/s по данни на NVIDIA).

==== 3.4.0.1 Контекст: от класически speculative decode към TiDAR

Преди TiDAR най-честият practical подход за ускорение е класически speculative decoding с draft + verify логика. Този подход е добра отправна точка, но често страда или от по-слаб draft модел, или от по-ниска ефективност на верификацията.

#figure(
  image("../../../images/Images_TiDAR_Optimization/Vanilla_speculative_decoding_with_smaller_model.png", width: 78%),
  caption: [Базов speculative decoding (референтен случай)],
) <fig-vanilla-spec>

В класическия вариант малък draft модел `q` и голям verify модел `p` работят с един и същ tokenizer и една и съща токенна азбука. Draft моделът предлага последователност от кандидати `x_1, ..., x_K`, а verify моделът ги проверява каузално за същия префикс.

$
alpha_i = min(1, frac(p_i(x_i), q_i(x_i)))
$

Тук `alpha_i` е вероятността за приемане на токена `x_i` на позиция `i`; ако токенът се отхвърли, се семплира от коригираща дистрибуция:

$
r_i(v) = frac(max(0, p_i(v) - q_i(v)), Z_i)
$

За KV cache е важно speculative токените да не се потвърждават окончателно предварително. Записва се само приетият префикс (или се прави rollback до приетата дължина), иначе cache състоянието се разминава с валидирания контекст и decode-ът деградира.

==== 3.4.0.2 Основна идея на TiDAR в един forward pass

TiDAR променя тази схема, като комбинира verify + predraft в един structured forward pass. Така се използва по-добре наличният паралелен compute и се намалява serving overhead-ът.

#figure(
  image("../../../images/Images_TiDAR_Optimization/TiDAR_Single_Forward_pass.png", width: 78%),
  caption: [TiDAR single-forward layout: verify + predraft (diagram from TiDAR paper)],
) <fig-tidar-single-forward>

На схемата verify частта оценява текущия draft, а predraft частта едновременно предлага следващите кандидати за идната итерация. Ключовият момент е, че това не са два отделни модела, а един и същ TiDAR модел, изпълнен веднъж с различно структурирани attention връзки в рамките на същия forward pass.

Така се спестява допълнителен model orchestration overhead (няма втори draft модел и допълнителен cross-model sync), а наличният GPU compute се използва по-плътно в една обща граф операция.

==== 3.4.0.3 Структурирани маски и достъп по позиции

Ключът е в маските: различни части от входа имат различен режим на внимание, така че едновременно да се пази AR логиката за "talk" и diffusion паралелизмът за "think".

#figure(
  image("../../../images/Images_TiDAR_Optimization/TiDAR_infernece_mask.png", width: 64%),
  caption: [Структурирана маска за TiDAR decode/предрафт логика (diagram from TiDAR paper)],
) <fig-tidar-decode-mask>

Verify токените виждат каузален контекст по същия принцип както при нормален speculative decoding. Predraft токените виждат останалите токени в своя predraft group с bidirectional attention и едновременно с това виждат префикса до позицията, след която трябва да предсказват, т.е. до съответния draft токен.

==== 3.4.0.4 Prefill маска

Prefill фазата подготвя началното KV-cache състояние за decode, като подава prompt контекста с коректна каузална видимост. Това гарантира, че последващите verify/predraft стъпки стъпват върху консистентна базова история.

От prefill изхода се взима и първият `D*` токен (първият draft токен), който служи като стартова точка за следващата decode итерация.

#figure(
  image("../../../images/TiDAR_prefill_mask.png", width: 64%),
  caption: [TiDAR prefill маска (diagram from TiDAR paper)],
) <fig-tidar-prefill-mask>

==== 3.4.0.5 Training маска

В training режим маската реализира едновременно AR и diffusion режим върху подравнени позиции, така че моделът да учи и следващ токен (causal), и паралелна denoising логика в diffusion частта.

Критичното предимство е, че training входът е с удвоен контекст `[clean | diff]` (приблизително `2S`), а не с decode-подобна експлозия от типа `K + K^2`. Така TiDAR може да се тренира с по-управляем memory/compute профил и със стандартен training pipeline за фиксирана контекстна дължина.

В paper-а е тествано corruption поведение с шум, маска и смесени режими; отчетено е, че вариантът само с mask corruption работи най-добре. Затова и в тази имплементация се използва единен `[MASK]` токен при diffusion частта на training/inference подредбата.

#figure(
  image("../../../images/TiDAR_training_mask.png", width: 52%),
  caption: [TiDAR training маска (diagram from TiDAR paper)],
) <fig-tidar-training-mask>

==== 3.4.0.6 Loss формулировка и баланс AR:Diff

В оригиналната формулировка training целта е:

$
L_T(omega) = frac(1, 1 + epsilon) (
  frac(epsilon, S) sum_(i=1)^S L_1(x_i, x_(i+1); omega)
  + frac(1, S) sum_(i=1)^S L_2(m, x_i; omega)
)
$

където `L_1` е AR терминът, `L_2` е diffusion терминът, `m` е `[MASK]` токенът, а `epsilon` контролира баланса между двата терма.

В paper-а е направен sweep за съотношението между двата термина и е наблюдавано, че баланс `1:1` (т.е. `alpha = beta = 1`) е най-стабилен и дава най-добър общ компромис. Тестваният диапазон е около `0.8 .. 1.2` за всеки коефициент.

В тази работа се приема същият принцип: AR и Diff компонентите се държат балансирани като базова настройка, а отклоненията се използват само при целеви експерименти.

==== 3.4.0.7 Anchor-TiDAR (финален акцент)

Anchor разширението въвежда стабилна референтна точка в decode стъпката и подобрява практическата ефективност на приемане/rollback логиката, особено при дълги генерации.

#figure(
  image("../../../images/Images_TiDAR_Optimization/Anchor_TiDAR_forward_pass.png", width: 78%),
  caption: [Anchor-TiDAR forward pass],
) <fig-anchor-forward>

Важно уточнение: Anchor-TiDAR е мое собствено разширение спрямо оригиналния TiDAR paper. В статията акцентът е по-скоро върху това моделът да е обучен достатъчно добре, така че пълен rejection (нула приети draft токени) да се случва минимално рядко, вместо да се описва изрично fallback механизъм за този случай. Ако все пак се случи full rejection, трябва или да се направи нов predraft pass, или да се държи допълнителен `K` набор (още compute), за да се избегне стоп в прогреса.

С Anchor подобрението се гарантира прогрес без нужда от допълнителен predraft compute в тази гранична ситуация, като целта е да се запази качеството на AR изхода. По-нататък е обяснено и как се управлява KV-cache pool-ът със static shape политика.

==== 3.4.1 Представяне на dual sequence вход
// COMPACT_CANDIDATE: 3.4.1 може да се съкрати при финален page-budget pass.

TiDAR training конструира вход с дължина `2S` във формат `[clean | diff]`:

- `clean` част (`1..S`) — стандартен AR поток за next-token предсказване;
- `diff` част (`S+1..2S`) — diffusion поток с `[MASK]` corruption и structured visibility.

На практика се учат две съгласувани задачи върху едни и същи таргети:

- AR: `x_i -> x_(i+1)` по каузален ред;
- Diff: `masked(x_i) -> x_i` в паралелен blockwise режим.

Това подравняване позволява директно сравнение и комбиниране на AR/Diff логити по позиции, което после се използва и при verify/predraft decode логиката.

Визуална референция за training маската: @fig-tidar-training-mask.

==== 3.4.2 Преобразуване и маскиране по позиции
// COMPACT_CANDIDATE: 3.4.2 може да се съкрати при финален page-budget pass.

`tidar_masks.build_tidar_train_bias` създава attention bias матрица, която управлява точно кои позиции се виждат в training pass-а. Ключовите правила са:

- clean -> clean: каузален достъп (класически AR);
- diff вътре в block: bidirectional достъп;
- diff -> clean prefix: разрешен достъп до нужния контекст;
- clean/diff към бъдещи невалидни позиции: забранен достъп.

Тези правила гарантират, че AR частта остава каузално коректна, а diffusion частта получава паралелна локална видимост без leakage към бъдеща информация извън допустимия контекст.

Decode/предрафт маската е показана по-горе на @fig-tidar-decode-mask.

==== 3.4.3 Извличане на verify и predraft логити
// COMPACT_CANDIDATE: 3.4.3 може да се съкрати при финален page-budget pass.

При inference входът се подрежда като `[current_draft | predraft_masks]`, след което от един forward pass се извличат:

- verify логити за текущия draft;
- K predraft предложения за следващата стъпка.

Технически това е важно, защото verify и predraft не се изпълняват като две отделни model извиквания. Логитите се взимат от различни позиционни срезове на един и същ output тензор, което:

- намалява host orchestration overhead;
- подобрява ефективността при static-shape изпълнение;
- поддържа консистентен KV-cache update модел за следващата итерация.

==== 3.4.4 KV-cache pointer commit/rollback

Anchor-TiDAR използва оптимистичен запис в KV cache и семантика с указател: приетият префикс се потвърждава чрез `prefix_len`, а отхвърленият суфикс се отстранява логически чрез връщане на указателя.

#figure(
  block(inset: 6pt, stroke: 0.6pt + rgb("#D1D5DB"), radius: 4pt)[
    #grid(
      columns: (43%, 57%),
      gutter: 8pt,
      align: top,
      [
        #align(left)[
          #text(size: 9pt, weight: "bold")[A) Поток на една Anchor-TiDAR decode итерация]
          #v(3pt)
          #diagram(
            cell-size: (16mm, 9mm),
            spacing: 6pt,
            node-stroke: 0.8pt + rgb("#4B5563"),
            edge-stroke: 0.8pt,
            mark-scale: 70%,

            node((0, 0), [Prefill
L = prompt len], width: 36mm, fill: rgb("#E5E7EB")),
            node((0, 1), [Single forward          \
verify + K predraft], width: 44mm, fill: rgb("#DBEAFE")),
            node((0, 2), [Acceptance check        \
anchor + verify], width: 40mm, fill: rgb("#FDE68A")),
            node((0, 3), [Accepted prefix = r     \
(1 <= r <= K)], width: 40mm, fill: rgb("#FECACA")),
            node((0, 4), [Pointer commit          \
L := L + r], width: 34mm, fill: rgb("#DCFCE7")),
            node((0, 5), [Next draft from         \
selected proposal], width: 40mm, fill: rgb("#E9D5FF")),

            edge((0, 0), (0, 1), "-|>"),
            edge((0, 1), (0, 2), "-|>"),
            edge((0, 2), (0, 3), "-|>"),
            edge((0, 3), (0, 4), "-|>"),
            edge((0, 4), (0, 5), "-|>"),
          )
        ]
      ],
      [
        #align(left)[
          #text(size: 9pt, weight: "bold")[B) KV cache pointer семантика (пример с частично приемане)]
          #v(2pt)

          #text(size: 7.9pt)[
            Преди стъпката:#linebreak()
            committed cache `A B C D* E F`,#linebreak()
            указател `L=6`.
          ]
          #v(2pt)
          #align(left)[#tok("A") #h(2pt) #tok("B") #h(2pt) #tok("C") #h(2pt) #tok("D*", fill: rgb("#FDE68A"), stroke_color: rgb("#DC2626")) #h(2pt) #tok("E", fill: rgb("#BFDBFE")) #h(2pt) #tok("F", fill: rgb("#BFDBFE"))]

          #v(3pt)
          #text(size: 7.9pt)[
            Decode pass изпълнява оптимистичен#linebreak()
            запис за текущия draft `G' H I`.#linebreak()
            Временно се достига `L+K=9`.
          ]
          #v(2pt)
          #align(left)[#tok("A") #h(2pt) #tok("B") #h(2pt) #tok("C") #h(2pt) #tok("D*", fill: rgb("#FDE68A"), stroke_color: rgb("#DC2626")) #h(2pt) #tok("E", fill: rgb("#BFDBFE")) #h(2pt) #tok("F", fill: rgb("#BFDBFE")) #h(2pt) #tok("G'", fill: rgb("#C4B5FD")) #h(2pt) #tok("H", fill: rgb("#C4B5FD")) #h(2pt) #tok("I", fill: rgb("#FCA5A5"), stroke_color: rgb("#B91C1C"))]

          #v(3pt)
          #text(size: 7.9pt)[
            При несъответствие в последната#linebreak()
            verify позиция (`I' != I`)#linebreak()
            се приема само префиксът `G' H`.#linebreak()
            Суфиксът `I` се отхвърля логически#linebreak()
            чрез pointer update.
          ]
          #v(2pt)
          #align(left)[#tok("A") #h(2pt) #tok("B") #h(2pt) #tok("C") #h(2pt) #tok("D*", fill: rgb("#FDE68A"), stroke_color: rgb("#DC2626")) #h(2pt) #tok("E", fill: rgb("#BFDBFE")) #h(2pt) #tok("F", fill: rgb("#BFDBFE")) #h(2pt) #tok("G'", fill: rgb("#C4B5FD"), stroke_color: rgb("#166534")) #h(2pt) #tok("H", fill: rgb("#C4B5FD"), stroke_color: rgb("#166534")) #h(2pt) #tok("I", fill: rgb("#F3F4F6"), stroke_color: rgb("#9CA3AF"))]
          #v(2pt)
          #text(size: 7.9pt)[
            Краен commit за стъпката: `L := 8`.#linebreak()
            Ефективният committed cache е#linebreak()
            `A B C D* E F G' H`.
          ]
        ]
      ],
    )
  ],
  caption: [KV cache commit/rollback при Anchor-TiDAR: поток и pointer update],
)

==== 3.4.5 Decode сценарий (Anchor-TiDAR)

1. `Prefill`: записва се prompt в KV cache и се получава стартовият `D*`.
2. `Forward pass`: едновременно се извличат verify логити и `K` predraft предложения.
3. `Acceptance`: предложенията се проверяват по ред и се приема най-дългият валиден префикс.
4. `Commit/Rollback`: KV pointer се премества само до приетата дължина.
5. `Next step`: от приетия префикс и новия anchor започва следващата итерация.

Визуални референции: @fig-anchor-forward и @fig-tidar-single-forward.

==== 3.4.6 Допълнителна визуална интерпретация за Mij

Обозначаваме с `A` anchor токена, с `D1..Dk` - draft токени, а с `Mij` - токен в predraft блок `i` на позиция `j`.

`Mij` вижда:

- каузално: префикса + anchor + нужния verify префикс;
- двупосочно: токените в собствения predraft блок;
- не вижда: нерелевантните бъдещи блокове извън текущата structured група.

#figure(
  block(inset: 6pt, stroke: 0.6pt + rgb("#D1D5DB"), radius: 4pt)[
    #align(left)[
    #grid(
      columns: (auto),
      row-gutter: 4pt,
      [#tok("A", fill: rgb("#FDE68A"), stroke_color: rgb("#DC2626")) #h(3pt) #tok("D1", fill: rgb("#BFDBFE"), stroke_color: rgb("#DC2626")) #h(3pt) #tok("D2", fill: rgb("#BFDBFE"), stroke_color: rgb("#DC2626")) #h(3pt) #tok("D3", fill: rgb("#BFDBFE")) #h(3pt) #tok("D4", fill: rgb("#BFDBFE"))],
      [#tok_pad() #h(3pt) #tok("M00", fill: rgb("#E9D5FF")) #h(3pt) #tok("M01", fill: rgb("#E9D5FF")) #h(3pt) #tok("M02", fill: rgb("#E9D5FF")) #h(3pt) #tok("M03", fill: rgb("#E9D5FF")) #h(3pt) #tok("M04", fill: rgb("#E9D5FF"))],
      [#tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok("M10", fill: rgb("#E9D5FF")) #h(3pt) #tok("M11", fill: rgb("#E9D5FF")) #h(3pt) #tok("M12", fill: rgb("#E9D5FF")) #h(3pt) #tok("M13", fill: rgb("#E9D5FF")) #h(3pt) #tok("M14", fill: rgb("#E9D5FF"))],
      [#tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok("M20", fill: rgb("#E9D5FF"), stroke_color: rgb("#DC2626")) #h(3pt) #tok("M21", fill: rgb("#C4B5FD"), stroke_color: rgb("#DC2626")) #h(3pt) #tok("M22", fill: rgb("#E9D5FF"), stroke_color: rgb("#DC2626")) #h(3pt) #tok("M23", fill: rgb("#E9D5FF"), stroke_color: rgb("#DC2626")) #h(3pt) #tok("M24", fill: rgb("#E9D5FF"), stroke_color: rgb("#DC2626"))],
      [#tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok("M30", fill: rgb("#E9D5FF")) #h(3pt) #tok("M31", fill: rgb("#E9D5FF")) #h(3pt) #tok("M32", fill: rgb("#E9D5FF")) #h(3pt) #tok("M33", fill: rgb("#E9D5FF")) #h(3pt) #tok("M34", fill: rgb("#E9D5FF"))],
      [#tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok_pad() #h(3pt) #tok("M40", fill: rgb("#E9D5FF")) #h(3pt) #tok("M41", fill: rgb("#E9D5FF")) #h(3pt) #tok("M42", fill: rgb("#E9D5FF")) #h(3pt) #tok("M43", fill: rgb("#E9D5FF")) #h(3pt) #tok("M44", fill: rgb("#E9D5FF"))],
    )
    #place(top + left, dx: 80pt, dy: 4pt)[
      #line(length: 52pt, angle: 90deg, stroke: 1pt + rgb("#DC2626"))
    ]
    ]
  ],
  caption: [Триъгълна визуализация на verify/predraft зависимостите; пример с маркиран `M21`],
)

При токен `M21`:

- каузално внимание: `prefix + A + D1 + D2`;
- двупосочно внимание: `M20..M24` (в рамките на собствения блок);
- липса на внимание към останалите предрафт блокове.

Сравнение с класически speculative decode: @fig-vanilla-spec.

=== 3.5 Експеримент: Free Token Slots

В този раздел се анализира явлението "Free Token Slots" - случаи, в които TiDAR може да използва допълнителна паралелна работа в една decode стъпка по-ефективно от класически AR decode път. Експериментите са върху A40 профили и са свързани директно с избора на structured masks, single-forward verify/predraft и KV-cache стратегията, описани по-горе.

В TiDAR paper-а има сходен тип анализ, но тук той е повторен върху значително по-слаб хардуер (A40 вместо H100), като е разширен и с допълнителни тестове за sampling режими и greedy decoding поведение.

Точно това явление е и основната мотивация зад начина, по който TiDAR е конструиран: да запълва "свободните" изчислителни слотове в decode стъпката с полезна verify/predraft работа, вместо GPU ресурсът да остава неизползван между последователните AR операции.

*Важно за интерпретацията:* показаните throughput резултати са benchmark-нати при зададен теоретичен `accept_rate = 0.8` (около 80% приети токени на итерация). За тези експериментални матрици не са тренирани отделни нови модели за всеки сценарий; сравняват се скоростни профили при фиксирана теоретична accuracy настройка.

==== 3.5.1 Native decode attention

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    align: top,
    [
      #fts.line-chart(
        "Native decode (1B-style, lower is better)",
        fts.n1b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
      )
    ],
    [
      #fts.line-chart(
        "Native decode (3B-style, lower is better)",
        fts.n3b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
      )
    ],
    [
      #fts.line-chart(
        "Native decode (7B-style, lower is better)",
        fts.n7b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
      )
    ],
  ),
  caption: [Free Token Slots - native decode attention при 1B, 3B, 7B (A40)],
)

Легенда (decode attention графики):

- #box(width: 8pt, height: 8pt, fill: fts.c-n-structured, radius: 1pt)[] #h(4pt) синя линия — TiDAR native structured (стандартният decode път със structured mask; почти припокрива се с червената);
- #box(width: 8pt, height: 8pt, fill: fts.c-n-dense, radius: 1pt)[] #h(4pt) червена линия — TiDAR native dense / no-mask variant (същият decode път, но без подаване на structured mask; долната от двете близки TiDAR линии);
- #box(width: 8pt, height: 8pt, fill: fts.c-ar, radius: 1pt)[] #h(4pt) лилава линия — AR baseline (`K x len1`, горната линия).

Наблюдението тук е, че TiDAR native decode кривите остават значително по-плоски спрямо AR baseline, който е мащабиран като `K x len1`. Това е practically важният сигнал за проекта: при нарастване на `K` се получава по-добра амортизация на compute разхода на стъпка и по-добро използване на наличния паралелен ресурс.

==== 3.5.2 Kernel terms (контролен анализ)

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    align: top,
    [
      #fts.line-chart(
        "Kernel terms (1B-style, lower is better)",
        fts.k1b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
      )
    ],
    [
      #fts.line-chart(
        "Kernel terms (3B-style, lower is better)",
        fts.k3b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
      )
    ],
    [
      #fts.line-chart(
        "Kernel terms (7B-style, lower is better)",
        fts.k7b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
      )
    ],
  ),
  caption: [Kernel terms експеримент за attention pass варианти (1B, 3B, 7B)],
)

Легенда (kernel terms графики):

- #box(width: 8pt, height: 8pt, fill: fts.c-k-structured, radius: 1pt)[] #h(4pt) синя линия — kernel structured (реално използваният вариант);
- #box(width: 8pt, height: 8pt, fill: fts.c-k-dense, radius: 1pt)[] #h(4pt) червена линия — kernel dense / no-mask (почти винаги най-долна);
- #box(width: 8pt, height: 8pt, fill: fts.c-k-densezero, radius: 1pt)[] #h(4pt) зелена линия — dense + jax-zero variant (при K=2,4 е най-долно, при K>4 започва да расте);
- #box(width: 8pt, height: 8pt, fill: fts.c-ar, radius: 1pt)[] #h(4pt) лилава линия — AR baseline (`K x len1`, консистентно най-горна права линия).

Тези графики са контролен експеримент за различни attention pass варианти и целят да изолират ефекта на маската върху compute профила. Важно: тук не се влиза в custom kernel посока (Pallas/Triton/собствени CUDA/C++ kernel-и); анализът е на текущия production-like decode път. Както и в paper-а, custom kernel оптимизациите остават по-скоро посока за бъдещо развитие.

==== 3.5.3 MLP scaling

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    align: top,
    [
      #fts.line-chart(
        "MLP only (1B-style, lower is better)",
        fts.mlp1b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
        y-label: "ms / MLP",
        y-label-dx: 6.5pt,
      )
    ],
    [
      #fts.line-chart(
        "MLP only (3B-style, lower is better)",
        fts.mlp3b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
        y-label: "ms / MLP",
        y-label-dx: 6.5pt,
      )
    ],
    [
      #fts.line-chart(
        "MLP only (7B-style, lower is better)",
        fts.mlp7b,
        width: 58mm,
        height: 40mm,
        x-label: "K",
        y-label: "ms / MLP",
        y-label-dx: 6.5pt,
      )
    ],
  ),
  caption: [Free Token Slots - MLP scaling сравнение (1B, 3B, 7B)],
)

Легенда (MLP графики):

- #box(width: 8pt, height: 8pt, fill: fts.c-mlp-tidar, radius: 1pt)[] #h(4pt) тюркоазена линия — TiDAR MLP при `L = K + K^2` (долната линия);
- #box(width: 8pt, height: 8pt, fill: fts.c-mlp-ar, radius: 1pt)[] #h(4pt) лилава линия — AR MLP baseline (`K x len1`, горната линия).

MLP графиките подсилват същата идея: при TiDAR токенният блок `L = K + K^2` се обработва в един общ pass, докато AR baseline се акумулира почти линейно с `K`. За текущата архитектура това е ключова връзка между теорията и практиката - structured single-model decode пътят не само работи коректно, а и носи реална throughput полза в настройките, върху които е разработен проектът.

==== 3.5.4 Sampling path (greedy и non-greedy)

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    align: top,
    [
      #fts.line-chart(
        "Sampling-only 500m (lower is better)",
        fts.samp500,
        width: 58mm,
        height: 45mm,
        x-label: "K",
        y-label: "ms / sampling cycle",
        y-label-dx: -8pt,
      )
    ],
    [
      #fts.line-chart(
        "Sampling-only 1B (lower is better)",
        fts.samp1b,
        width: 58mm,
        height: 45mm,
        x-label: "K",
        y-label: "ms / sampling cycle",
        y-label-dx: -8pt,
      )
    ],
    [
      #fts.line-chart(
        "Sampling-only 3B (lower is better)",
        fts.samp3b,
        width: 58mm,
        height: 45mm,
        x-label: "K",
        y-label: "ms / sampling cycle",
        y-label-dx: -8pt,
      )
    ],
  ),
  caption: [Free Token Slots - sampling-only сравнение (500m, 1B, 3B)],
)

Легенда (sampling графики):

- #box(width: 8pt, height: 8pt, fill: fts.c-samp-ti-g, radius: 1pt)[] #h(4pt) тюркоазена линия — TiDAR staged sampling, greedy (`top_k=0`);
- #box(width: 8pt, height: 8pt, fill: fts.c-samp-ar-g, radius: 1pt)[] #h(4pt) лилава линия — AR scaled baseline, greedy (`K x len1`);
- #box(width: 8pt, height: 8pt, fill: fts.c-samp-ti-ng, radius: 1pt)[] #h(4pt) жълта линия — TiDAR staged sampling, non-greedy (`top_k=50`);
- #box(width: 8pt, height: 8pt, fill: fts.c-samp-ar-ng, radius: 1pt)[] #h(4pt) червена линия — AR scaled baseline, non-greedy (`top_k=50`).

Sampling-only резултатите показват, че при greedy режим TiDAR държи по-нисък sampling cycle почти по целия диапазон на `K`. При non-greedy (`top_k=50`) има crossover поведение: при ниски `K` разликата е малка/в полза на AR, а при по-високи `K` TiDAR започва да печели, което е в синхрон с целта да се използва по-добре паралелният compute при по-широк draft.

Как работи staged sampling в този benchmark: първо се семплира verify токенът за текущата позиция, след което се прави rejection/select стъпка за predraft предложенията и се семплира само избраният predraft ред за следващата итерация. Така се мери изолирано именно sampling/rejection пътят (без transformer forward), което позволява директно сравнение с AR `K x len1` baseline.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    align: top,
    [
      #fts.delta-bar-chart(
        "Staged sampling speedup (%) - top_k=0 (higher is better)",
        fts.micro-topk0,
        bar-color: fts.c-micro-0,
        width: 84mm,
        height: 34mm,
        title-size: 8.4pt,
      )
    ],
    [
      #fts.delta-bar-chart(
        "Staged sampling speedup (%) - top_k=50 (higher is better)",
        fts.micro-topk50,
        bar-color: fts.c-micro-50,
        width: 84mm,
        height: 34mm,
        title-size: 8.4pt,
      )
    ],
  ),
  caption: [Изолиран sampling speedup: staged path спрямо old path],
)

Практически извод: в тези run-ове sampling частта е малък дял от целия decode wall-time (приблизително под ~1%). Това се вижда от факта, че изолираният sampling speedup е голям, но end-to-end decode подобрението остава около десети от процента; следователно основният bottleneck остава transformer forward пътят, а не самото sampling действие. Въпреки това free token slots остават важни, защото показват къде има неизползван паралелен ресурс и къде следващите оптимизации могат да донесат допълнителен throughput.

=== 3.6 Резултати: TiDAR срещу AR (A40 и H100)

След анализа на Free Token Slots тук показваме head-to-head резултатите за TiDAR срещу AR при еднаква 3b конфигурация върху A40 и H100.

#figure(
  grid(
    columns: (58%, 42%),
    gutter: 8pt,
    align: top,
    [
      #hh.line-chart(
        "Speedup vs AR by prefill len",
        hh.overlay_speedup_lines,
        width: 84mm,
        height: 48mm,
      )
    ],
    [
      *Легенда:*

      - #box(width: 8pt, height: 8pt, fill: hh.c-h100-k16, radius: 1pt)[] #h(4pt) H100, K=16;
      - #box(width: 8pt, height: 8pt, fill: hh.c-a40-k16, radius: 1pt)[] #h(4pt) A40, K=16;
      - #box(width: 8pt, height: 8pt, fill: hh.c-h100-k8, radius: 1pt)[] #h(4pt) H100, K=8;
      - #box(width: 8pt, height: 8pt, fill: hh.c-a40-k8, radius: 1pt)[] #h(4pt) A40, K=8.
    ],
  ),
  caption: [Head-to-head: speedup vs AR по prefill (A40 + H100)],
)

#figure(
  hh.bar-chart(
    "Speedup vs AR (steady TPS)",
    hh.overlay_bars,
    width: 178mm,
    height: 52mm,
  ),
  caption: [Head-to-head: обобщена speedup диаграма (full width)],
)

Тези резултати потвърждават, че наблюдаваните ускорения не са единичен артефакт от една карта, а се запазват и при по-висок клас GPU, като при `K=16` H100 показва най-силна speedup крива.

=== 3.7 Разширена loss функция за TiDAR

Тази секция описва разширената loss формулировка в имплементацията, при която отделните обучителни термини са с независими коефициенти и могат да се комбинират контролирано според целта на конкретния training stage.

==== 3.7.1 Формулировка с независими коефициенти

В практическата имплементация замених нормализацията от paper-а `/(1 + alpha)` с модулна сума от отделни термини, всеки със собствен коефициент. Така всеки компонент може да се усилва, отслабва или да се изключва с нулев коефициент, което прави възможни чисти ablation експерименти и целенасочени комбинации между различни обучителни сигнали.

$
L = alpha L_(A) + beta L_(D) + rho D_(f) + chi D_(r) + delta L_(H) + delta_(m) L_(P) + eta L_(S) + gamma L_(K)
$

Където `bar(P_(A))` означава AR разпределение със stop-gradient (без обратен градиент към AR клона).

- `AR` термин:

  $
  L_(A) = -frac(1, S - 1) sum_(t=0)^(S-2) log P_(A,t)(x_(t+1))
  $

  Поддържа стандартната next-token езикова способност в clean половината.

- `Diff` термин:

  $
  L_(D) = -frac(1, S - 1) sum_(t=0)^(S-2) log Q_(D,t)(x_(t+1))
  $

  Учи diffusion половината да реконструира същата бъдеща цел от mask-нат вход.

- Forward KL термин:

  $
  D_(f) = sum_v bar(P_(A)(v)) log frac(bar(P_(A)(v)), Q_(D)(v))
  $

  Наказва Diff, когато изпуска вероятностна маса, която AR счита за важна.

- Reverse KL термин:

  $
  D_(r) = sum_v Q_(D)(v) log frac(Q_(D)(v), bar(P_(A)(v)))
  $

  Наказва Diff, когато разпределя излишна маса извън AR разпределението.

- Hard agreement термин:

  $
  L_(H) = -log Q_(D)(v^*)
  $

  Притиска greedy избора на Diff да съвпада с greedy избора на AR; `v*` е токенът с най-висока вероятност според AR.

- Masked hard agreement термин (prefix mask):

  $
  L_(P) = frac(1, N_(P)) sum_(i in P) -log Q_(D,i)(v^*_(A,i))
  $

  Това е новият "маскиран" hard loss: взима същия hard сигнал, но само върху позициите до първото несъвпадение между AR и Diff (включително). В кода този термин се управлява с отделен коефициент `delta_masked` и служи да фокусира натиска върху префикса, който е най-важен за accept логиката.

- Soft distillation термин:

  $
  L_(S) = T^2 D(p_(A)^T || p_(D)^T)
  $

  Пренася "меката" форма на AR разпределението към Diff, без да се фиксира само един токен; `p^T` са температурно-скалирани вероятности.

- Top-K set distillation термин:

  $
  L_(K) = -log sum_(v in V_K) Q_(D)(v)
  $

  Концентрира вероятностната маса на Diff в топ-K набора на AR и подобрява шансa за accept при decode; `V_K` е множеството от топ-K AR кандидати.

Важно техническо поведение в кода: при коефициент `0` съответният термин се пропуска напълно от изчислението (`alpha`, `beta`, `rho`, `chi`, `delta`, `delta_masked`, `eta`, `gamma`), което позволява експериментите да се правят с минимален излишен compute и с ясна изолация на ефекта от всеки компонент.

Механизъм на JAX компилацията: в `TiDAR/model/Run_training.py` функцията `_run_chunk` е JIT-компилирана чрез `@partial(jax.jit, static_argnames=...)`, като loss коефициентите са подадени като static аргументи. При нова комбинация от стойности се компилира нов изпълним вариант, а при повторение на същата комбинация се използва кешираният вариант. Тъй като branch-овете в `TiDAR/model/Training_step.py` са условни по коефициент, изключените термини се prune-ват от съответния jaxpr.

=== 3.8 Резултати от greedy run-ове (135M vs 360M)

Тази секция обобщава резултати от реално тренирани TiDAR варианти с `draft_len=8`: 135M и 360M. Данните са от продължителен run (около 5 часа), при който по-големият модел е стартиран с по-голям batch, а логовете са подравнени по optimizer стъпки.

Легенда за сравняваните модели:

- #box(width: 8pt, height: 8pt, fill: rg.color-135, radius: 1pt)[] #h(4pt) 135M модел - run на RTX 5090 (32GB), платена цена за наем 5.23 USD;
- #box(width: 8pt, height: 8pt, fill: rg.color-360, radius: 1pt)[] #h(4pt) 360M модел - run на RTX PRO 6000 (96GB), платена цена за наем 11.81 USD.
- #box(width: 8pt, height: 8pt, fill: rg.color-delta, radius: 1pt)[] #h(4pt) сива линия = делта `135M / 360M` (1.0 означава равни стойности).

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    row-gutter: 8pt,
    align: top,
    [
      #rg.compare-chart(
        (
          (label: "135M", color: rg.color-135, data: rg.ar-135),
          (label: "360M", color: rg.color-360, data: rg.ar-360),
        ),
        width: 84mm,
        height: 42mm,
        x_label: "Optimizer стъпка",
        y_label: "AR (lower is better)",
        delta_data: rg.delta-ratio-series(rg.ar-135, rg.ar-360),
      )
    ],
    [
      #rg.compare-chart(
        (
          (label: "135M", color: rg.color-135, data: rg.diff-135),
          (label: "360M", color: rg.color-360, data: rg.diff-360),
        ),
        width: 84mm,
        height: 42mm,
        x_label: "Optimizer стъпка",
        y_label: "Diff (lower is better)",
        delta_data: rg.delta-ratio-series(rg.diff-135, rg.diff-360),
      )
    ],
    [
      #rg.compare-chart(
        (
          (label: "135M", color: rg.color-135, data: rg.hard-135),
          (label: "360M", color: rg.color-360, data: rg.hard-360),
        ),
        width: 84mm,
        height: 42mm,
        x_label: "Optimizer стъпка",
        y_label: "Hard (lower is better)",
        delta_data: rg.delta-ratio-series(rg.hard-135, rg.hard-360),
      )
    ],
    [
      #rg.compare-chart(
        (
          (label: "135M", color: rg.color-135, data: rg.acc-135),
          (label: "360M", color: rg.color-360, data: rg.acc-360),
        ),
        width: 84mm,
        height: 42mm,
        x_label: "Optimizer стъпка",
        y_label: "Accept (higher is better)",
        delta_data: rg.delta-ratio-series(rg.acc-135, rg.acc-360),
      )
    ],
  ),
  caption: [Обучителни метрики: 135M срещу 360M],
)

Практически извод: 360M вариантът държи по-стабилни и по-ниски Diff/Hard загуби и по-висок greedy acceptance, но идва с осезаемо по-висока цена във време и ресурс. Моделът е приблизително 2.66 пъти по-голям по брой параметри спрямо 135M, а в реалните run-ове обучението му беше около 3-4 пъти по-бавно. Тези диаграми показват и че началните способности на базовия модел директно се отразяват върху ученето на TiDAR future prediction-ите: по-силният стартов модел учи по-стабилно тази задача, което е консистентно с по-големия му капацитет и с факта, че е предварително трениран върху повече данни. За проекта това е важен ориентир за trade-off между качество, цена и време за итерация при TiDAR training.

=== 3.9 Проведен експеримент с различни loss функции и конфигурации

В този раздел е представен контролиран експеримент за TiDAR post-training с множество loss конфигурации. Основната цел е да се оцени кои комбинации подобряват ключовите метрики (`Diffusion loss` и `Greedy acceptance`), като едновременно с това се запази качеството на оригиналния AR модел (без значимо влошаване на AR loss спрямо последните итерации от базовото GIANT обучение). Дизайнът е избран така, че да поддържа бърз експериментален цикъл: промяна на loss конфигурация, кратък run, метрикa-анализ и следваща итерация.

==== 3.9.0 Избор на TinyStories и базова конфигурация

За експерименталната серия е избран TinyStories, защото е сравнително малък корпус с ниска ентропия. Това позволява бързи итерации и ясна интерпретация на ефекта от loss промените. В практиката този тип корпус се научава стабилно и от базов модел около `30M` параметъра, без да е необходимо пълно минаване през целия набор във всеки run.

В конкретната серия базовият GIANT модел (`30M` параметъра) е обучен върху около `1.05` милиарда токена за приблизително `33` минути на RTX 5090 (`0.89 USD/час`), а всеки TiDAR post-training run е отнемал около `2.5` часа на същия хардуер. Общата експериментална кампания е проведена в рамките на една седмица с междинен анализ след всеки run (включително логитни хистограми), при приблизителен общ разход около `32 USD`.

Базовият GIANT модел в тази серия е дефиниран със следната конфигурация:

#cli_block[
```yaml
model:
  embedding_size: 384
  num_heads: 6
  num_kv_heads: 2
  num_layers: 18
  feed_forward_size: 1024
  context_length: 512
training:
  batch_size: 32
  gradient_accumulation: 4
stages:
  - dataset: tinystories_512
    seq_len: 512
    epochs: 2
    fraction: 0.75
```
]

След базовото AR обучение се преминава към TiDAR post-training със същата архитектура, като началният режим е `alpha=1, beta=1`:

#cli_block[
```yaml
tidar:
  draft_length: 6
training:
  batch_size: 16
  gradient_accumulation: 4
  loss:
    alpha: 1.0
    beta: 1.0
    rho: 0.0
    chi: 0.0
    delta: 0.0
    eta: 0.0
stages:
  - dataset: tinystories_512
    seq_len: 512
    epochs: 3
    fraction: 0.75
```
]

Примерни изречения/промпт фрагменти от корпуса:

- "There was a small village."
- "Once upon a time, a little girl..."
- "In the forest, a tiny fox..."
- "One day, the teacher said..."
- "The robot wanted to learn..."

==== 3.9.1 Експериментален протокол и избор на checkpoint

Основният TiDAR baseline run (`alpha=1, beta=1`) достига `121566` optimizer стъпки. За сравнителната част е избран checkpoint около `90k`, от който се пускат над 10 различни loss конфигурации до приблизително `121k`, за да се измери ефектът им върху Diffusion/Greedy метриките при близки стартови условия.

Изборът на `90k` е целенасочен: в този етап baseline режимът започва видимо да забавя нормалното си учене при `alpha=beta=1`, което го прави подходяща точка за branch сравнения. Част от експериментите са стартирани и от нулева стъпка за допълнителна проверка на ранната динамика.

Branch-based дизайнът е силен за сравнение, защото:

- елиминира ефекта от различна ранна инициализация;
- сравнява режимите върху близка checkpoint основа;
- позволява директни delta криви спрямо baseline по една и съща step ос.

Във файла са използвани два прозореца на анализ:

- пълен прозорец (за стабилния run) `0-121k` за глобална динамика;
- zoom прозорец `90k-121k` за head-to-head сравнение между branch вариантите.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    [
      #tsc.multi-line-chart(
        ((color: tsc.stable-color, data: tsc.stable-ar),),
        width: 84mm,
        height: 44mm,
        y_label: "AR loss",
        y_ticks: 4,
        y_tick_decimals: 1,
        cut_x: tsc.cut-step,
        cut_label: "90k",
        x_min: tsc.stable-x-min,
        x_max: tsc.branch-x-max,
      )
    ],
    [
      #move(dx: 4pt)[
        #tsc.multi-line-chart(
          ((color: tsc.stable-color, data: tsc.stable-greedy),),
          width: 84mm,
          height: 44mm,
          y_label: "Greedy acc",
          y_ticks: 4,
          y_tick_decimals: 0,
          y_tick_percent: true,
          cut_x: tsc.cut-step,
          cut_label: "90k",
          x_min: tsc.stable-x-min,
          x_max: tsc.branch-x-max,
          y_min: tsc.greedy-y-min,
          y_max: tsc.greedy-y-max,
        )
      ]
    ],
  ),
  caption: [Базова траектория и branch cut при 90k стъпка],
)

==== 3.9.2 Сравнявани loss режими

Легендата по-долу обобщава сравняваните конфигурации в sweep-а. Тя покрива както "меки" alignment сигнали (KL/Distill), така и "твърди" acceptance-ориентирани сигнали (delta, delta_masked, top-k set), което е методологично правилно за TiDAR, където целта не е само нисък loss, а по-ефективен speculative decode.

Легенда на вариантите:

- #box(width: 8pt, height: 8pt, fill: tsc.stable-color, radius: 1pt)[] #h(4pt) Stable - `alpha=1, beta=1`, референтна линия;
- #box(width: 8pt, height: 8pt, fill: tsc.kl-only-color, radius: 1pt)[] #h(4pt) KL-only - `alpha=0.2, beta=0, rho=0.5, delta=0.3`, без Diff CE;
- #box(width: 8pt, height: 8pt, fill: tsc.kl-keep-color, radius: 1pt)[] #h(4pt) KL-keep - `alpha=1, beta=1, rho=0.1, delta=0.3`, умерен alignment;
- #box(width: 8pt, height: 8pt, fill: tsc.distill-color, radius: 1pt)[] #h(4pt) Distill - `alpha=1, beta=1, eta=0.04, T=2`, мека дистилация AR->Diff;
- #box(width: 8pt, height: 8pt, fill: tsc.smallar-color, radius: 1pt)[] #h(4pt) Greedy eta - агресивен agreement/дистилация режим;
- #box(width: 8pt, height: 8pt, fill: tsc.biggerbeta-color, radius: 1pt)[] #h(4pt) Bigger beta - `alpha=1, beta=5`, усилен Diff натиск;
- #box(width: 8pt, height: 8pt, fill: tsc.topk-color, radius: 1pt)[] #h(4pt) Top-K set - `gamma=0.01, gamma_topk=8`, късен stage;
- #box(width: 8pt, height: 8pt, fill: tsc.biggamma-color, radius: 1pt)[] #h(4pt) Big Gamma+Delta - силен Top-K + hard agreement;
- #box(width: 8pt, height: 8pt, fill: tsc.maskedlater-color, radius: 1pt)[] #h(4pt) Masked Delta later - частичен 90k+ run с `delta_masked`;
- #box(width: 8pt, height: 8pt, fill: tsc.deltamasked-color, radius: 1pt)[] #h(4pt) Delta masked early - кратък ранен run под 30k.

Run-ове, които водят до твърде бърза деградация на AR quality (catastrophic forgetting) или показват незначим ефект спрямо основната група, не са включени във финалното сравнение в тази секция.

==== 3.9.3 Интерпретация на AR и Diff кривите

AR панелите показват, че стабилният режим и умерените добавки (KL-keep, Distill) запазват близка и устойчива AR динамика в `90k-121k`. Това означава, че тези варианти не разрушават основната next-token способност.

Diff панелите дават по-силен разделителен сигнал:

- при KL-only (`beta=0`) Diff loss след `90k` става нефункционален като сравним индикатор (затова е изключен от zoom Diff панела);
- KL-keep и Distill остават в сравним диапазон със stable, което е знак, че дифузионният клон продължава да учи полезно, докато се добавя alignment.

Изводът е, че пълното изключване на Diff CE може да повиши някои acceptance метрики краткосрочно, но отслабва контролa върху самата Diff реконструкция и прави режима по-рисков за дълги run-ове.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    [
      #tsc.multi-line-chart(
        (
          (color: tsc.stable-color, data: tsc.stable-ar-90),
          (color: tsc.kl-only-color, data: tsc.kl-only-ar-90),
          (color: tsc.kl-keep-color, data: tsc.kl-keep-ar-90),
          (color: tsc.distill-color, data: tsc.distill-ar-90),
        ),
        width: 84mm,
        height: 44mm,
        y_label: "AR",
        x_min: tsc.branch-x-min,
        x_max: tsc.branch-x-max,
        y_min: tsc.ar-zoom-y-min,
        y_max: tsc.ar-zoom-y-max,
      )
    ],
    [
      #tsc.multi-line-chart(
        (
          (color: tsc.stable-color, data: tsc.stable-diff-90),
          (color: tsc.kl-keep-color, data: tsc.kl-keep-diff-90),
          (color: tsc.distill-color, data: tsc.distill-diff-90),
        ),
        width: 84mm,
        height: 44mm,
        y_label: "Diff",
        x_min: tsc.branch-x-min,
        x_max: tsc.branch-x-max,
        y_min: tsc.diff-zoom-y-min,
        y_max: tsc.diff-zoom-y-max,
      )
    ],
  ),
  caption: [AR и Diff zoom сравнение за 90k-121k],
)

#figure(
  tsc.multi-line-chart(
    (
      (color: tsc.stable-color, data: tsc.stable-diff),
      (color: tsc.deltamasked-color, data: tsc.deltamasked-diff),
    ),
    width: 170mm,
    height: 46mm,
    y_label: "Diff",
    cut_x: tsc.cut-step,
    cut_label: "90k",
    x_min: tsc.stable-x-min,
    x_max: tsc.branch-x-max,
  ),
  caption: [Пълен Diff прозорец: stable срещу ранен delta_masked run],
)

==== 3.9.4 Greedy acceptance: основен таргет на sweep-а

Greedy acceptance е централната метрика в документа (`argmax(AR) == argmax(Diff)` по валидни позиции). Показаните панели имат три нива на интерпретация:

- absolute панели (90k-121k): сравнение на реалните траектории;
- delta панели спрямо stable: видимост дали вариантът е устойчиво над/под baseline;
- extended панели: добавят агресивни режими и по-късни експериментални хипотези.

Най-важните наблюдения за практиката са:

- KL-keep и Distill дават по-добър баланс между acceptance печалба и запазена training стабилност;
- KL-only може да покаже моментни плюсове в greedy, но на цената на "изключен" Diff обучителен сигнал;
- по-агресивните режими (напр. Big Gamma+Delta) изискват внимателно stage планиране, защото увеличават чувствителността към хиперпараметри;
- `delta_masked` (късен и ранен вариант) е концептуално обещаващ, защото натиска точно acceptance-критичния префикс, но частичният характер на част от run-овете налага предпазлива интерпретация.

Част от run-овете са прекратявани по-рано, когато се наблюдава силно покачване на AR loss (catastrophic forgetting) или когато Diffusion/Greedy метриките тръгват устойчиво под основната група. Поради това някои криви са по-къси и не достигат до последните точки от общия прозорец. Всички логове и checkpoint-и от тези run-ове са запазени в репото.

#figure(
  tsc.multi-line-chart(
    (
      (color: tsc.stable-color, data: tsc.stable-greedy-90),
      (color: tsc.kl-only-color, data: tsc.kl-only-greedy-90),
      (color: tsc.kl-keep-color, data: tsc.kl-keep-greedy-90),
      (color: tsc.distill-color, data: tsc.distill-greedy-90),
    ),
    width: 170mm,
    height: 46mm,
    y_label: "Greedy",
    x_min: tsc.branch-x-min,
    x_max: tsc.branch-x-max,
    y_min: tsc.greedy-y-min,
    y_max: tsc.greedy-y-max,
  ),
  caption: [Greedy acceptance: core варианти в 90k-121k],
)

#figure(
  tsc.multi-line-chart(
    (
      (color: tsc.kl-only-color, data: tsc.delta-greedy-kl-only),
      (color: tsc.kl-keep-color, data: tsc.delta-greedy-kl-keep),
      (color: tsc.distill-color, data: tsc.delta-greedy-distill),
    ),
    width: 170mm,
    height: 46mm,
    y_label: "Delta",
    x_min: tsc.branch-x-min,
    x_max: tsc.branch-x-max,
    y_min: tsc.delta-greedy-y-min,
    y_max: tsc.delta-greedy-y-max,
    baseline_y: 0.0,
  ),
  caption: [Delta срещу stable за core вариантите],
)

#figure(
  tsc.multi-line-chart(
    (
      (color: tsc.stable-color, data: tsc.stable-greedy-90),
      (color: tsc.kl-only-color, data: tsc.kl-only-greedy-90),
      (color: tsc.kl-keep-color, data: tsc.kl-keep-greedy-90),
      (color: tsc.distill-color, data: tsc.distill-greedy-90),
      (color: tsc.smallar-color, data: tsc.smallar-greedy-90),
      (color: tsc.biggerbeta-color, data: tsc.biggerbeta-greedy-90),
      (color: tsc.topk-color, data: tsc.topk-greedy-90),
      (color: tsc.biggamma-color, data: tsc.biggamma-greedy-90),
      (color: tsc.maskedlater-color, data: tsc.maskedlater-greedy-90),
    ),
    width: 170mm,
    height: 48mm,
    y_label: "Greedy",
    x_min: tsc.branch-x-min,
    x_max: tsc.branch-x-max,
    y_min: tsc.greedy-ext-y-min,
    y_max: tsc.greedy-ext-y-max,
  ),
  caption: [Greedy acceptance: разширен набор от всички варианти],
)

#figure(
  tsc.multi-line-chart(
    (
      (color: tsc.kl-only-color, data: tsc.delta-greedy-kl-only),
      (color: tsc.kl-keep-color, data: tsc.delta-greedy-kl-keep),
      (color: tsc.distill-color, data: tsc.delta-greedy-distill),
      (color: tsc.smallar-color, data: tsc.delta-greedy-smallar),
      (color: tsc.biggerbeta-color, data: tsc.delta-greedy-biggerbeta),
      (color: tsc.topk-color, data: tsc.delta-greedy-topk),
      (color: tsc.biggamma-color, data: tsc.delta-greedy-biggamma),
      (color: tsc.maskedlater-color, data: tsc.delta-greedy-maskedlater),
    ),
    width: 170mm,
    height: 58mm,
    y_label: "Delta",
    x_min: tsc.branch-x-min,
    x_max: tsc.branch-x-max,
    y_min: tsc.delta-greedy-ext-y-min,
    y_max: tsc.delta-greedy-ext-y-max,
    baseline_y: 0.0,
  ),
  caption: [Delta срещу stable за разширения набор],
)

==== 3.9.5 Inference резултати (Accept/Iter таблица)

Секцията "Inference Results" проверява дали training ефектите се пренасят при реален greedy decode (`temperature=0`, `draft_len=6`) върху 10 промпта и три дължини на генериране (50/100/300 стъпки). Ключовите тенденции в таблицата са:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    [Checkpoint], [Steps=50], [Steps=100], [Steps=300],
    [Stable \@90k], [2.41 / #text(fill: rgb("#C62828"), weight: "bold")[3.50] / 1.63], [2.26 / 2.75 / 1.71], [2.10 / 2.45 / #text(fill: rgb("#C62828"), weight: "bold")[1.88]],
    [Stable \@121k], [2.22 / 3.27 / 1.53], [2.09 / 2.68 / 1.60], [2.10 / 2.69 / 1.57],
    [KL-only \@121k], [#text(fill: rgb("#C62828"), weight: "bold")[2.42] / 3.27 / #text(fill: rgb("#C62828"), weight: "bold")[1.81]], [2.30 / 2.68 / 1.80], [2.08 / 2.49 / 1.61],
    [KL-keep \@121k], [2.40 / 3.27 / 1.75], [2.29 / 2.83 / 1.80], [2.18 / 2.90 / 1.75],
    [Distill \@121k], [2.31 / #text(fill: rgb("#C62828"), weight: "bold")[3.50] / 1.53], [2.26 / #text(fill: rgb("#C62828"), weight: "bold")[3.09] / 1.87], [#text(fill: rgb("#C62828"), weight: "bold")[2.19] / 2.74 / 1.80],
    [SmallAR+eta \@105k], [2.15 / 2.88 / 1.75], [2.15 / 2.75 / #text(fill: rgb("#C62828"), weight: "bold")[1.90]], [2.00 / 2.43 / 1.74],
    [Bigger beta \@121.5k], [2.37 / 3.27 / 1.63], [2.33 / #text(fill: rgb("#C62828"), weight: "bold")[3.09] / 1.62], [1.96 / 2.27 / 1.74],
    [Top-K set \@121k], [2.28 / 3.12 / 1.52], [2.18 / 2.56 / 1.72], [#text(fill: rgb("#C62828"), weight: "bold")[2.19] / #text(fill: rgb("#C62828"), weight: "bold")[2.91] / 1.56],
    [Big Gamma+Delta \@105k], [2.32 / 3.33 / 1.67], [2.26 / 2.86 / 1.79], [2.09 / 2.59 / 1.71],
    [Masked Delta later \@105k], [2.30 / 3.12 / 1.72], [#text(fill: rgb("#C62828"), weight: "bold")[2.34] / 2.86 / 1.72], [2.01 / 2.40 / 1.67],
  ),
  caption: [Accept/Iter summary върху 10 TinyStories промпта],
)

- при `Steps=300` най-силен среден резултат дават Distill (`2.19`) и Top-K set (`2.19`), следвани от KL-keep (`2.18`), над Stable (`2.10`);
- при `Steps=100` Stable bigger beta достига висока средна стойност (`2.33`), но при дългия хоризонт (`300`) пада до `1.96`, което подсказва по-слабa устойчивост;
- SmallAR big greedy eta и Masked Delta later са по-агресивни режими и остават под най-добрите устойчиви варианти при по-дълги генерации.

Това потвърждава работната хипотеза, че умереният alignment (KL-keep или Distill) е по-надежден за обща decode ефективност от прекалено силни наказания/насърчения върху отделен компонент.

==== 3.9.6 Локален механистичен анализ чрез logits хистограми

Логитните хистограми дават позиционен анализ на механизма за accept/reject и допълват агрегираните метрики с локална причинна интерпретация. Наблюдава се следният типичен модел:

- в ранните позиции на draft блока AR и Diff често са подравнени по top-1 и токените се приемат;
- около семантично по-нееднозначни позиции Diff става по-разфокусиран или измества top-1 и там започват rejection-и;
- точно тези позиции обясняват защо prefix-ориентирани loss-и (като `delta_masked`) имат смисъл: те таргетират момента, в който acceptance веригата се прекъсва.

Този тип визуализация е силен аргумент, че sweep-ът не е "black-box" оптимизация по една метрика, а контролирано търсене на причинно обясними подобрения.

#pagebreak()

#tsc.render-hist(tsc.meta1, tsc.data1)

#v(6pt)

#tsc.render-hist(tsc.meta2, tsc.data2)

#v(6pt)

#tsc.render-hist(tsc.meta3, tsc.data3)

#pagebreak()

==== 3.9.7 Финален на експеримента и план за продължение

Проведеният експеримент изпълнява основната си цел: да валидира end-to-end TiDAR training/inference pipeline и да даде първа сравнителна картина за ефекта от различни loss конфигурации. В същото време резултатите показват, че при текущата постановка разделителната способност между вариантите е ограничена.

Основното наблюдение е силното припокриване на кривите между различните директории/конфигурации. В значителна част от диапазона разликите са минимални и на места практически неразличими визуално. Това е индикация, че в текущия режим доминира влиянието на корпуса, а не на конкретния избор на auxiliary loss.

Работната интерпретация е, че TinyStories е нискоентропиен и силно структуриран корпус. При такава среда локална грешка в отделна позиция често не води до трайно разминаване по следващите токени, защото контекстът остава лесен за възстановяване. Поради това режими като `delta_masked` могат да изглеждат по-слаби в ранния диапазон (`0-30k`), без това задължително да означава по-нисък потенциал на самия метод в по-труден домейн.

Втори ключов фактор е мащабната разлика спрямо оригиналната експериментална среда на NVIDIA. Тук базовият модел и тренировъчният корпус са приблизително 50 пъти по-малки, докато референтната постановка използва значително по-голям предварително обучен модел (Qwen 1.5B) и много по-широка предтренировъчна база. Това ограничение директно намалява чувствителността на експеримента към фини ефекти от loss комбинациите.

Планирано продължение до защитата (следващи ~3 месеца):

- втори експериментален цикъл върху по-информативен корпус от типа OpenWebText/Wikipedia/куриран web text;
- увеличение на модела и тренировъчния обем, така че loss режимите да се разграничат по-ясно;
- избор на най-устойчивите конфигурации от този междинен етап и пренасяне към финален training диапазон от порядъка `300M-1B` параметри и `5B-30B` токена.

Следователно текущият експеримент се приема като междинен, но необходим етап: системата е валидирана, наблюдавани са ограниченията на малък/лесен корпус, и е дефиниран конкретен план за доразвитие преди финалната защита.
