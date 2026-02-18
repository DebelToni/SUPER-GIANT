= ЧЕТВЪРТА ГЛАВА

== РЪКОВОДСТВО НА ПОТРЕБИТЕЛЯ

=== 4.1 Инсталация

Базовият начин за стартиране е чрез Docker образите в `CICD/Docker/`.

Минимален сценарий:

```bash
docker run --pull always -d --gpus all \
  --name giant-training \
  --mount type=bind,source="$HOME/GIANT",target=/proj \
  bonanc/giant-training:latest

docker exec -it giant-training bash
```

За automatic-S3-sync workflow е наличен отделен образ `CICD/Docker/giant-training-S3/`, за който е нужна база като Minio или подобна в облака, за постоянно обновяване на данните. При слаб интернет от страна на developer машината това е непрепоръчително. S3 tool-овете са достатъчни за базовия вариант.

=== 4.2 Стартиране на средата и създаване на експеримент

Под "проект" в тази система се разбира комбинация от:

- глобална конфигурация (`GIANT/v2/Global_Config.yml`, `TiDAR/Global_Config.yml`);
- локална model/training конфигурация (`Config.yml`);
- директории за dataset/checkpoints/logs.

Първата стъпка е избор на data root и checkpoint root, след което се изпълняват pipeline и training скриптовете.

=== 4.3 Подготовка и импорт на datasets

Импортиране на datasets.

Примери:

```bash
python GIANT/v2/data_pipeline/build_corpus.py --config GIANT/v2/data_pipeline/Config.yml
python TiDAR/data_pipeline/Run_pipeline.py --config TiDAR/data_pipeline/data_configs/Greedy_exp_500m.yml
```

Системата ще генерира Arrow shard-ове и manifest/statistics файлове.

=== 4.4 Работа с training stages

Stage структурата играе ролята на curriculum по време на обучение.

==== 4.4.1 Добавяне и пренареждане на stages

Еквивалент: добавяне и пренареждане на `stages` в YAML. Всеки stage може да задава `seq_len`, `epochs`, `fraction` и stage-local loss коефициенти.

==== 4.4.2 Избор на checkpoint

Еквивалент: избор на конкретна checkpoint версия за инференс.

```bash
python GIANT/v2/model/Generate_faster.py --checkpoint latest
python TiDAR/model/inference.py --checkpoint /proj/giant-data/TiDAR/checkpoints/params/step_0012100.npz
```

==== 4.4.3 Разделяне на run на етапи

Еквивалент: разделяне на training run на етапи с различни objectives или подмяна на config по време на следващ stage.

==== 4.4.4 Нулиране на експеримент

Еквивалент: reset на конкретен експеримент чрез нов checkpoint root или презаписване на stage output директория (pipeline регенерация).

=== 4.5 Управление на изпълнението

Основни команди за runtime управление:

- старт на обучение: `Run_training.py`;
- пауза/спиране: `SIGINT`/`SIGTERM` (graceful stop след текущ chunk);
- resume: `--resume latest` или път до конкретна checkpoint;
- inference режими: greedy (`--temperature 0`) или sampling (`--temperature > 0`, `--top_k`).

=== 4.6 Контрол на KV cache и inference мащаб

За инференс производителност `Generate_faster.py` поддържа KV cache bucket избор:

- автоматичен избор;
- конфигурационни buckets;
- CLI override чрез `--kv_cache_buckets`;
- изключване чрез `--disable_kv_buckets`.

Това позволява баланс между памет и latency, особено при различни prompt/steps комбинации.

=== 4.7 Настройка на training параметри

Настройка на hyperparameters и loss коефициенти.

За TiDAR основните контроли са в `training.loss`:

- `alpha`, `beta`;
- `rho`, `chi`;
- `delta`, `delta_masked`;
- `eta`, `eta_T`;
- `gamma`, `gamma_topk`.

=== 4.8 Клавишни комбинации и бързи команди

Работният процес е CLI-first. Типични бързи команди:

- `python GIANT/v2/model/Run_training.py --resume latest`
- `python GIANT/v2/model/Evaluate.py --checkpoint latest`
- `python GIANT/v2/model/Chat.py --chat --steps 128`
- `python GIANT/v2/model/Generate_faster.py --prompt "Hello" --steps 64 --verbose`
- `python TiDAR/model/Run_training.py --config TiDAR/model/Config_135m.yml`
- `python TiDAR/model/inference.py --prompt "Hello" --draft_len 8 --temperature 0.0`

#figure(
  image("../../../images/Docker.png", width: 62%),
  caption: [Примерен Docker-базиран стартов workflow за системата],
)
