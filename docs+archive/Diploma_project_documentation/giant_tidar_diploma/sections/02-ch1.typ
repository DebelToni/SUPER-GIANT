= ПЪРВА ГЛАВА

== ПРЕГЛЕД НА СЪЩЕСТВУВАЩИ СИСТЕМИ, ПОДОБНИ НА GIANT, И ИЗВЕСТНИ РАЗВОЙНИ СРЕДСТВА

В тази глава са разгледани решения и инструменти, близки до тематиката на дипломната работа: обучение и ускорен инференс на decoder-only езикови модели.

=== LLaMA семейство (Meta)

LLaMA моделите се утвърдиха като де факто стандарт за отворени foundation модели. Основните им предимства са добра scaling стратегия, стабилни архитектурни избори (RMSNorm, RoPE, SwiGLU) и голяма екосистема от производни реализации. За проекта GIANT тези модели са референтна точка за архитектурни и обучителни практики.

=== SmolLM / компактни open модели

SmolLM профилите са важни като practical baseline за експерименти с ограничен GPU бюджет. В TiDAR частта на проекта се използват точно такива размери (135M и 360M), което позволява многократни сравнения между loss конфигурации и decode варианти в разумно време.

=== nanoGPT / minGPT

Тези проекти показват минимална, ясно четима реализация на autoregressive Transformer. Те са полезни за концептуално разбиране, но не покриват production-like нуждите от sharding, resume, инфраструктура за отдалечен GPU и систематични benchmark отчети.

=== Megatron-LM и distributed тренировка

Megatron-LM е ориентир за мащабно обучение с моделна/данна паралелизация. В настоящата работа не се гони multi-node distributed setup; фокусът е единичен GPU workflow с максимална инженерна практичност. Въпреки това принципите за разделяне на data/model/optimizer отговорности са сходни.

=== vLLM и TensorRT-LLM

Тези системи са водещи при serving/latency оптимизации и силно ефективно управление на KV-cache. В GIANT/TiDAR част от идеите за cache политика и throughput benchmarking са приложени в custom JAX стек, като изследователският фокус е върху Anchor-TiDAR декодиране и loss-driven подобрения.

=== Python

Изборът на Python е определен от бързия цикъл за изследователска разработка, широка библиотечна поддръжка и добра интеграция с ML инструменти, cloud execution и автоматизация.

=== JAX + Flax

JAX е избран заради XLA компилацията, контрол върху устройството и възможност за високопроизводителни JIT графи. Flax осигурява структурирано model definition API и variable collections, критични за KV-cache и за чисто разделяне на train/inference състояние.

=== Optax + Orbax

Optax е използван за optimizer pipeline (clip + AdamW + scheduler), а Orbax - за асинхронни mini checkpoint-и. Тази комбинация подобрява устойчивостта при long-running експерименти и намалява риска от загуба на прогрес при прекъсване.

=== HuggingFace Hub / Datasets / Transformers

Data pipeline частта използва `datasets` за ingest на публични корпуси (streaming и non-streaming режими), а `transformers` - за tokenizer интеграция. Това позволява стабилен и възпроизводим preprocess workflow към Arrow шардове.

=== Docker, Tailscale и S3 инструменти

Инфраструктурният слой в `CICD/` дава:

- Docker образи с готов CUDA/JAX стек;
- Tailscale-based secure remote достъп;
- S3 операции чрез `s5cmd` wrapper;
- скриптове за бърз sync на локални промени към remote GPU среда.

#figure(
  image("../../../images/Docker.png", width: 66%),
  caption: [Контейнерна среда за reproducible изпълнение],
)
