#import "_common.typ": placeholder_figure

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

==== 3.2.2 TrainLoopOrchestrator

`Run_training.py` реализира scan-базиран training loop с gradient accumulation, finite checks и checkpoint политика.

===== 3.2.2.1 Основен JIT training loop (`_run_chunk` + `lax.scan`)

Сърцето на изпълнението е `_run_chunk(...)`, където batch-овете се обработват със `lax.scan`, пресмятат се loss/grad стойности и се правят проверки за non-finite стъпки.

===== 3.2.2.2 Обработка на сигнали и контролирано спиране

Скриптът обработва `SIGINT/SIGTERM` и спира контролирано след текущия chunk, за да не се загуби междинно състояние.

===== 3.2.2.3 Gradient accumulation и update политика

Обучението поддържа `gradient_accumulation`; update се прилага само при достигане на зададения accumulation праг, а остатъчните градиенти се доизчистват при край на stage.

===== 3.2.2.4 Периодично логване на метрики

На всеки `log_every` стъпки се записват `global_step`, `stage`, `loss` и допълнителни показатели (напр. `ppl`, `accept`, `greedy_acc`, KL/hard/distill/top-k при TiDAR).

===== 3.2.2.5 Resume от checkpoint и dataloader state

Възстановяването зарежда параметри, optimizer state и dataloader прогрес (`stage_index`, `stage_step_total`, `stage_states`), така че обучението да продължи от последната консистентна точка.

==== 3.2.3 DataLoaderManager

`ShardedArrowDataset` + `StageDataLoader` реализират shard-aware четене, shuffle, batching и state restore.

==== 3.2.4 CheckpointManager

`checkpoint_manager.py` капсулира запис/зареждане на параметри, optimizer state и metadata.

===== 3.2.4.1 Метод SaveParameters

Запис на `.npz` checkpoint с атомарно `tmp -> rename` поведение.

===== 3.2.4.2 Метод SaveOptimizerState

Паралелен запис на optimizer буфери (`msgpack`) за коректен resume.

===== 3.2.4.3 Метод RestoreLatest

Намиране на последната достъпна checkpoint стъпка и зареждане на съответното състояние.

===== 3.2.4.4 Метод SetMetadata

Запис на метаданни за quality показатели (`val_loss`, `val_ppl`, `train_loss`).

===== 3.2.4.5 Метод AsyncMiniCheckpoint

Мини checkpoint-и в background режим чрез Orbax за по-нисък риск при preemption.

==== 3.2.5 InferenceManager

Inference слоят покрива prefill, decode, chat режим, KV bucket политика и throughput измерване.

===== 3.2.5.1 Метод Prefill

`jit_inference.prefill` запълва KV cache от prompt и връща начално състояние за decode.

===== 3.2.5.2 Метод Decode

`jit_inference.decode` използва `lax.scan` за генериране на нови токени със sampling или greedy стратегия.

===== 3.2.5.3 Метод KVBucketSelect

`Generate_faster.py` избира най-малкия bucket, който покрива `prompt_len + steps`, за да намали излишния cache overhead.

===== 3.2.5.4 Метод ChatTurn

`Chat.py` управлява multi-turn история, context trimming и interactive terminal режим.

===== 3.2.5.5 Метод BatchedPrefill

`test_batched_inference.py` валидира batched prefill с отмествания и проверява съвпадение с per-sample baseline.

===== 3.2.5.6 Метод StopOnEos

Генерацията може да прекъсва на EOS токен за контрол на отговора и избягване на нежелано продължение.

===== 3.2.5.7 Метод ThroughputReport

Отчетите измерват prefill/decode време, tokens/s и policy сравнения по контекст/модел/драфт параметри.

==== 3.2.6 RemoteOpsManager

`CICD/tools/` и Docker setup покриват deployment и remote execution.

===== 3.2.6.1 Основен принцип на работа

Синхронизация на локални промени, remote run в стандартизирана среда и съхранение на артефакти в споделен storage.

===== 3.2.6.2 Метод GetSnappedFrame

Избор на консистентна remote state точка преди старт на training/inference run.

=== 3.3 Реализация на data pipeline

==== 3.3.1 Документ като източник на токени

Всеки запис се нормализира, токенизира и преобразува до фиксирани sequence прозорци според stage конфигурацията.

==== 3.3.2 Преобразуване от документ към sequence прозорци

`SequenceEmitter` реализира short/long document логика, random/sequential windowing и flush на remainder.

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

==== 3.3.6 Обработка на chat маски

При chat datasets се изгражда assistant-aware маска, така че loss да се натрупва само върху желаните роли.

===== 3.3.6.1 Нормализация

Поддържат се NFKC и whitespace normalization, приложими по stage.

===== 3.3.6.2 Дедупликация

Опционална дедупликация работи с hash-based схема и ограничение на ключовете.

===== 3.3.6.3 JSON/JSONL поддръжка

Поддържат се incremental JSON четене и line-wise JSONL parsing.

===== 3.3.6.4 Стабилност

Пайплайнът записва атомарно и може да се преизпълнява stage-по-stage.

==== 3.3.7 Освобождаване на ресурси и стабилност

Системата използва defensive проверки за невалидни батчове и non-finite стойности, за да предотврати разпространение на грешки в параметрите.

=== 3.4 Реализация на TiDAR обучение и decode синхронизация

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
