#import "_common.typ": placeholder_figure
#import "../../../../TiDAR/Docs/Bucket_Prefix0_1600_A40_Spot.typ": line-chart, c-red, c-blue, c-orange, c-green, m500_steady, m500_full_steady, m500_ar_steady, m3b_steady, m3b_full_steady, m3b_ar_steady

#let tok(lbl, fill: rgb("#E5E7EB")) = box(
  inset: (x: 4pt, y: 2pt),
  stroke: 0.5pt + rgb("#374151"),
  fill: fill,
  radius: 2pt,
)[#text(size: 7.5pt)[#lbl]]

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
    width: 42%,
    inset: 8pt,
    radius: 5pt,
    stroke: 0.6pt + rgb("#9CA3AF"),
    fill: luma(242),
  )[
```text
128
###.....................
###.....................
###.....................

256
######..................
######..................
######..................

512
############............
############............
############............

1024
########################
########################
########################
```
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
        "A40 spot: 500m / k=4 (bucketed vs full vs AR)",
        (
          (label: "500m bucketed", color: c-blue, data: m500_steady),
          (label: "500m full-context", color: c-red, data: m500_full_steady),
          (label: "500m AR", color: c-green, data: m500_ar_steady),
        ),
        width: 86mm,
        height: 42mm,
      )
    ],
    [
      #line-chart(
        "A40 spot: 3b / k=16 (bucketed vs full vs AR)",
        (
          (label: "3b bucketed", color: c-orange, data: m3b_steady),
          (label: "3b full-context", color: c-red, data: m3b_full_steady),
          (label: "3b AR", color: c-green, data: m3b_ar_steady),
        ),
        width: 86mm,
        height: 42mm,
      )
    ],
  ),
  caption: [A40 throughput криви: 500m/k=4 и 3b/k=16],
)

Легенда на линиите:

- #box(width: 8pt, height: 8pt, fill: c-blue, radius: 1pt)[] #h(4pt) синя линия — 500m bucketed (най-горна в лявата графика в повечето точки);
- #box(width: 8pt, height: 8pt, fill: c-red, radius: 1pt)[] #h(4pt) червена линия — TiDAR full-context baseline (средна в лявата графика);
- #box(width: 8pt, height: 8pt, fill: c-green, radius: 1pt)[] #h(4pt) зелена линия — AR baseline (най-долна в лявата графика);
- #box(width: 8pt, height: 8pt, fill: c-orange, radius: 1pt)[] #h(4pt) оранжева линия — 3b bucketed (най-горна в дясната графика в повечето точки).

Метриката е steady decode tokens/s (*high is better*).

В тези диаграми sweep-ът е с генерация от `0` до `1600` токена. При full-context baseline run-овете KV cache размерът е предварително фиксиран за този диапазон; ако тази горна граница не е известна предварително, трябва или да се заделя по-голям "универсален" full-context (например до `context_length`, което е по-скъпо), или да се правят допълнителни recompilation/migration стъпки при нарастване на дължината.

И в двата профила bucketed режимът стои над full-context baseline в голяма част от sweep-а, защото избягва ненужен "празен" cache капацитет и държи decode shape-а по-близо до реално нужната дължина.

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

Преди TiDAR, най-честият practical подход за ускорение е класически speculative decoding с draft + verify логика. Този подход е добра отправна точка, но често страда или от по-слаб draft модел, или от по-ниска ефективност на верификацията.

#figure(
  image("../../../images/Images_TiDAR_Optimization/Vanilla_speculative_decoding_with_smaller_model.png", width: 78%),
  caption: [Базов speculative decoding (референтен случай)],
)

В класическия вариант малък draft модел `q` и голям verify модел `p` работят с един и същ tokenizer и една и съща токенна азбука. Draft моделът предлага последователност от кандидати `x_1, ..., x_K`, а verify моделът ги проверява каузално за същия префикс.

$
alpha_i = min(1, frac(p_i(x_i), q_i(x_i)))
$

Тук `alpha_i` е вероятността за приемане на токена `x_i` на позиция `i`; ако токенът се отхвърли, се семплира от коригираща дистрибуция:

$
r_i(v) = frac(max(0, p_i(v) - q_i(v)), Z_i)
$

За KV cache е важно speculative токените да не се commit-ват окончателно предварително. Записва се само приетият префикс (или се прави rollback до приетата дължина), иначе cache състоянието се разминава с валидирания контекст и decode-ът деградира.

==== 3.4.0.2 Основна идея на TiDAR в един forward pass

TiDAR променя тази схема, като комбинира verify + predraft в един structured forward pass. Така се използва по-добре наличният паралелен compute и се намалява serving overhead-ът.

#figure(
  image("../../../images/Images_TiDAR_Optimization/TiDAR_Single_Forward_pass.png", width: 78%),
  caption: [TiDAR single-forward layout: verify + predraft],
)

==== 3.4.0.3 Структурирани маски и достъп по позиции

Ключът е в маските: различни части от входа имат различен режим на внимание, така че едновременно да се пази AR логиката за "talk" и diffusion паралелизмът за "think".

#figure(
  image("../../../images/Images_TiDAR_Optimization/TiDAR_infernece_mask.png", width: 78%),
  caption: [Структурирана маска за TiDAR decode/предрафт логика],
)

==== 3.4.0.4 Anchor-TiDAR (финален акцент)

Anchor разширението въвежда стабилна референтна точка в decode стъпката и подобрява практическата ефективност на приемане/rollback логиката, особено при дълги генерации.

#figure(
  image("../../../images/Images_TiDAR_Optimization/Anchor_TiDAR_forward_pass.png", width: 78%),
  caption: [Anchor-TiDAR forward pass],
)

==== 3.4.1 Представяне на dual sequence вход

TiDAR training конструира вход `[clean | diff]`, където clean половината следва AR режим, а diff половината използва blockwise bidirectional mask.

==== 3.4.2 Преобразуване и маскиране по позиции

`tidar_masks.build_tidar_train_bias` реализира правила за достъп между clean/diff токени и block boundaries.

#figure(
  image("../../../images/Images_TiDAR_Optimization/TiDAR_infernece_mask.png", width: 78%),
  caption: [Маска за TiDAR decode/предрафт логика],
)

==== 3.4.3 Извличане на verify и predraft логити

При inference входът се подрежда като `[current_draft | predraft_masks]`, след което от един forward pass се извличат:

- verify логити за текущия draft;
- K predraft предложения за следващата стъпка.

==== 3.4.4 KV-cache pointer commit/rollback

Anchor-TiDAR използва optimistic KV write и pointer semantics: приеманата част се commit-ва чрез `prefix_len`, а отхвърленият суфикс се "rollback-ва" логически чрез връщане на pointer-а.

#figure(
  image("../../../images/Images_TiDAR_Optimization/Anchor_TiDAR_KV_cache_per_interation.png", width: 80%),
  caption: [KV cache развитие по итерации при Anchor-TiDAR],
)

==== 3.4.5 Подробен decode пример (Anchor-TiDAR)

```text
Example of how my TiDAR variant would work in decode

Prefill Input -> Output after sample
ABC MMM  -> BCD* DEF
Current KV cache: A B C

Decode step 1 input:
D*EF MMM MMM MMM

Decode step 1 output after sampling:
E*F'G' EFG FGH GHI
Current KV cache: A B C D* E F

Now we check if E* is E from the draft on the input (assume success). Then we check F' to F of the input (assume success). That means we accept the last proposal GHI.

Decode step 2 input:
(note here we will take G' that we sampled form F on last step and replace it in the GHI block)
G'HI MMM MMM MMM

Decode step 2 output after sampling:
H*I'J' HIJ IJK JKL
Current KV cache: A B C D* E F G' H I

Now we check that I' matches the output from I in the input but for example I'!=I at the input. So we select proposal 2 which is IJK

! Here we did not accept the full draft so we need ot move the pointer that says up to where we have KV cache. Right now it says we have 9 KV caches written but because we did not accept I!=I' we need to bring back the pointer 1 step back. So its current value will be 8 and the cache would contain A B C D* E F G' H

Decode step 3 input:
(Here we take I' from the sampled from last step instead of the I that is in the IJK block).
I*JK
```

#figure(
  image("../../../images/Images_TiDAR_Optimization/Anchor_TiDAR_forward_pass.png", width: 78%),
  caption: [Илюстрация на Anchor-TiDAR single-forward decode стъпка],
)

#figure(
  image("../../../images/Images_TiDAR_Optimization/TiDAR_Single_Forward_pass.png", width: 78%),
  caption: [Single-forward layout за TiDAR verify + predraft],
)

==== 3.4.6 Допълнителна визуална интерпретация за Mij

```text
Additional Verbose explenation of how decode pass works:

At inference draft token Mij from block i sees the prefix tokens + also the anchor token + any currently verified token up to token i (including it) from the draft that is being validated from last step

So if we verify this:
A - anchor
D - drafts but starting from 1, becasue 0 is the anchor
Mij - Mask in theoretical draft i, at index j
```

#figure(
  block(inset: 6pt, stroke: 0.6pt + rgb("#D1D5DB"), radius: 4pt)[
    #grid(
      columns: (auto),
      row-gutter: 4pt,
      [#tok("A", fill: rgb("#FDE68A")) #h(3pt) #tok("D1", fill: rgb("#BFDBFE")) #h(3pt) #tok("D2", fill: rgb("#BFDBFE")) #h(3pt) #tok("D3", fill: rgb("#BFDBFE")) #h(3pt) #tok("D4", fill: rgb("#BFDBFE"))],
      [#h(18pt) #tok("M00", fill: rgb("#E9D5FF")) #h(3pt) #tok("M01", fill: rgb("#E9D5FF")) #h(3pt) #tok("M02", fill: rgb("#E9D5FF")) #h(3pt) #tok("M03", fill: rgb("#E9D5FF")) #h(3pt) #tok("M04", fill: rgb("#E9D5FF"))],
      [#h(36pt) #tok("M10", fill: rgb("#E9D5FF")) #h(3pt) #tok("M11", fill: rgb("#E9D5FF")) #h(3pt) #tok("M12", fill: rgb("#E9D5FF")) #h(3pt) #tok("M13", fill: rgb("#E9D5FF")) #h(3pt) #tok("M14", fill: rgb("#E9D5FF"))],
      [#h(54pt) #tok("M20", fill: rgb("#E9D5FF")) #h(3pt) #tok("M21", fill: rgb("#C4B5FD")) #h(3pt) #tok("M22", fill: rgb("#E9D5FF")) #h(3pt) #tok("M23", fill: rgb("#E9D5FF")) #h(3pt) #tok("M24", fill: rgb("#E9D5FF"))],
      [#h(72pt) #tok("M30", fill: rgb("#E9D5FF")) #h(3pt) #tok("M31", fill: rgb("#E9D5FF")) #h(3pt) #tok("M32", fill: rgb("#E9D5FF")) #h(3pt) #tok("M33", fill: rgb("#E9D5FF")) #h(3pt) #tok("M34", fill: rgb("#E9D5FF"))],
      [#h(90pt) #tok("M40", fill: rgb("#E9D5FF")) #h(3pt) #tok("M41", fill: rgb("#E9D5FF")) #h(3pt) #tok("M42", fill: rgb("#E9D5FF")) #h(3pt) #tok("M43", fill: rgb("#E9D5FF")) #h(3pt) #tok("M44", fill: rgb("#E9D5FF"))],
    )
  ],
  caption: [Triangular visualize на verify/predraft зависимостите; пример с маркиран M21],
)

При токен `M21`:

- каузално внимание: `prefix + A + D1 + D2`;
- двупосочно внимание: `M20..M24` (в рамките на собствения блок);
- липса на внимание към останалите предрафт блокове.

#figure(
  image("../../../images/Images_TiDAR_Optimization/Vanilla_speculative_decoding_with_smaller_model.png", width: 78%),
  caption: [Базов speculative decoding (референтен случай за сравнение)],
)

Отчетите за:

- свободни token slots на A40;
- throughput head-to-head A40 срещу H100;
- KV cache policy сравнения;
- дълги decode sweep-ове;

са вградени като оригинален Typst код в `Приложение Б`.
