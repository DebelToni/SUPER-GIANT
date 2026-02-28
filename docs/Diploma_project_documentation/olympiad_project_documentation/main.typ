#set page(width: 210mm, height: 297mm, margin: 20mm)
#set text(font: "Times New Roman", size: 12pt)
#set par(justify: true, leading: 1.5em)
#set heading(numbering: none)
#let page_footer = context align(center)[#text(size: 10pt)[#counter(page).display()]]
#set page(footer: none)

#import "../../../TiDAR/Docs/TiDAR_AR_A40_H100_HeadToHead.typ" as hh
#import "../../../TiDAR/Docs/TinyStories_TiDAR_losses_Comparison.typ" as tsc
#import "../../../TiDAR/Docs/Bucket_Prefix0_1600_A40_Spot.typ" as bp
#import "../../../TiDAR/Docs/KV_Cache_Policy_A40.typ" as kvp
#import "../../../TiDAR/Docs/Finding_Free_token_slots_A40.typ" as fts
#import "../../../TiDAR/Docs/Results_Greedy_runs_135_360.typ" as rg
#import "@preview/fletcher:0.5.8": diagram, node, edge

#let tok(lbl, fill: rgb("#E5E7EB"), stroke_color: rgb("#374151")) = box(
  width: 22pt,
  height: 12pt,
  inset: (x: 4pt, y: 2pt),
  stroke: 0.5pt + stroke_color,
  fill: fill,
  radius: 2pt,
)[#align(center)[#text(size: 7.5pt)[#lbl]]]

#let leg(label, color, size: 7.6pt) = [
  #box(width: 6pt, height: 6pt, fill: color, radius: 1pt)[]
  #h(2pt)
  #text(size: size)[#label]
]

#show heading.where(level: 1): it => [
  #v(0.5em)
  #it
  #v(0.6em)
]

#show heading.where(level: 2): it => [
  #v(0.4em)
  #it
  #v(0.5em)
]

#align(center)[
  #v(12em)
  #text(size: 22pt, weight: "bold")[ДОКУМЕНТАЦИЯ НА ПРОЕКТ]
  #v(3.6em)
  #text(size: 18pt, weight: "bold")[GIANT]
  #v(0.8em)
  #text(size: 13pt)[Система за подготовка, трениране и ускорен инференс на езикови модели]
  #v(4em)
  #text(size: 13pt, weight: "bold")[Автор]
  #text(size: 12pt)[Антон Иванов Христов]
  #v(1.4em)
  #text(size: 13pt, weight: "bold")[Ръководител]
  #text(size: 12pt)[Евгени Драгомиров Димов]
  #v(6em)
  #text(size: 12pt)[София, 2026]
]

#pagebreak()
#set page(footer: page_footer)

= I. ТЕМА

Темата на проекта е разработване на цялостна система за създаване и използване на езикови модели - от подготовката на данните и обучението до ускорен inference и количествена оценка на производителността. Реализацията обединява стабилна инженерна основа за GPT-style езиков модел (GIANT) и изследователски слой за бърза генерация (TiDAR/Anchor-TiDAR) в единен codebase.

Фокусът е върху decoder-only Transformer модели, KV cache оптимизации и сравнение между класически autoregressive decode и хибриден decode режим. В хибридния режим diffusion компонентът се реализира като итеративнo подобряване на латентна репрезентация на бъдещето с последваща AR верификация в един и същ forward pass. Целта е практическа система с възпроизводимо обучение, ускорен inference и измерими метрики за speedup/latency.

#block(
  width: 100%,
  inset: 8pt,
  stroke: 0.7pt + rgb("#9CA3AF"),
  radius: 4pt,
  fill: luma(244),
)[
  #text(weight: "semibold")[Пълната документация и най-новия код на проекта може да откриете на: ]#link("https://debeltoni.github.io/SUPER-GIANT")[debeltoni.github.io/SUPER-GIANT]
]

= II. АВТОРИ

*Антон Иванов Христов*

- ЕГН: 0751296460
- Телефон: 0887776969
- Имейл: `anton.i.hristov.2021@elsys-bg.org`
- Училище: Технологично училище "Електронни системи" към ТУ - София
- Клас: 12

= III. РЪКОВОДИТЕЛ

*Евгени Драгомиров Димов*

- Телефон: 0886694906
- Имейл: `evgeni.dimov@ocado.com`
- Длъжност: Софтуерен инженер, Ocado Technologies
- Роля в проекта: научен ръководител

#pagebreak()

= IV. РЕЗЮМЕ

Проектът е структуриран в два слоя с ясни роли.

*GIANT* е базовата инженерна платформа: data pipeline, decoder-only training loop, checkpoint/resume механизъм, ускорен inference с KV cache и стандартизирани benchmark сценарии. Този слой осигурява стабилен, възпроизводим и практически използваем моделeн pipeline.

*TiDAR/Anchor-TiDAR* е изследователското разширение върху същата инфраструктурапо вдъхновение от научния труд на NVIDIA Research - Think in Diffusion, talk in AutoRegression. То въвежда sequence-level hybrid decode (draft + verify в един forward pass), структурирани attention маски и допълнителни loss конфигурации за анализ на speed/quality поведението.

Комбинацията между двата слоя позволява едновременно production-oriented експлоатация и научно-изследователска работа в една и съща codebase, без дублиране на инструменти и без отделни несъвместими среди.

== 1. Цели

Основната цел е изграждане на работещ и разширяем LLM stack, който да даде измерими резултати в три направления: качество на модела, скорост на inference и възпроизводимост на експериментите.

Подцелите са:

- да се реализира data pipeline за подготовка на големи текстови корпуси (ingest, cleaning, sharding, indexing);
- да се реализира надежден training loop с resume и checkpoint recovery;
- да се изгради inference слой с KV cache optimization и измерване на speed/latency;
- да се валидира Anchor-TiDAR подходът с фокус върху Free Token Slots и speedup;
- да се изведе практическа методология за сравнение на decode режими при различен хардуер и различни model scales.

Кратък анализ на потребностите показва, че в наличните open-source решения често липсва интеграция между training и high-quality benchmarking pipeline в единна среда. Обикновено има добри отделни компоненти, но не и обща, reproducible, end-to-end структура с лесна поддръжка на експериментален слой. Настоящият проект адресира именно тази празнина.

== 2. Основни етапи в реализирането на проекта

Проектът е реализиран от един автор, поради което всички дейности са изпълнявани последователно в единна инженерна рамка:

1. Дефиниране на архитектурни изисквания и структури на конфигурации.
2. Изграждане на data pipeline и тренировъчни datasets.
3. Реализация на training pipeline за decoder-only Transformer.
4. Реализация на inference pipeline с KV cache optimization.
5. Добавяне на TiDAR/Anchor-TiDAR изследователски модул.
6. Провеждане на benchmark кампания и анализ на резултатите.
7. Документиране, верификация и подготовка на финални артефакти.

Ролята на научния ръководител е консултативна: насочване към по-добри инженерни практики, критерии за валидиране на резултатите и структуриране на експерименталната методология.

== 3. Ниво на сложност на проекта

Нивото на сложност е високо, защото системата комбинира няколко трудни слоя едновременно:

- *Numerical complexity* - работа с големи тензори, mixed precision режими, JAX/XLA compilation constraints и memory-bound behavior.
- *System complexity* - синхронизация между local CPU workflow и remote GPU execution, reproducible environments чрез Docker, надеждна работа с артефакти.
- *Algorithmic complexity* - едновременно поддържане на класически autoregressive decode и sequence-level hybrid decode с допълнителна masking логика.
- *Evaluation complexity* - коректно измерване на speedup, latency, acceptance и quality indicators без да се смесват compile overhead, warmup ефекти и run-to-run variance.

Проектът решава практически значим проблем: как един и същ модел да се използва едновременно за високо качество и за висок inference throughput, без задължителна зависимост от втори "draft" модел и без нарушаване на устойчивия инженерeн цикъл.

== 4. Логическо и функционално описание на решението

Системата е модулна и е организирана в два главни слоя:

- *GIANT Core* - стабилен training/inference stack.
- *TiDAR Extension* - изследователски слой за hybrid decode и допълнителни loss/mask механизми.

Взаимодействието между модулите е организирано чрез shared конфигурации, стандартизирани entry points и общ artifact storage модел.

=== 4.1. Базова архитектура на модела

#figure(
  grid(
    columns: (28%, 72%),
    gutter: 10pt,
    align: top,
    [#image("../../images/DecoderTransformer.png", width: 100%)],
    [
      Decoder-only архитектурата е организирана като последователност от attention + feed-forward блокове с residual връзки и нормализация между тях.

      Входният текст се преобразува в embedding представяния, след което преминава през поредица от causal Transformer блокове. На изхода се получават logits върху речника, които чрез softmax формират вероятности за следващ токен.

      При training обработката е паралелна в рамките на контекстния прозорец, а при inference се използва prefill + decode режим с KV cache.
    ],
  ),
  caption: [Decoder-only Transformer архитектура, използвана като базов модел],
)

Базовият модел е decoder-only Transformer с causal masking. Всеки training sample се обработва като последователност от токени, а целевата функция е next-token prediction. Архитектурно са използвани съвременни компоненти като RMSNorm, RoPE и SwiGLU, което осигурява стабилност при обучение и добра ефективност при inference.

Multi-head attention операторът е:

$ h_h = op("softmax")( frac((Q W_h^(Q)) (K W_h^(K))^T, sqrt(d_k)) + M ) (V W_h^(V)) $

$ op("MHA")(Q, K, V) = [h_1 || h_2 || ... || h_H] W^(O) $

където $M$ е причинно-следствена маска, която гарантира, че позиция $i$ вижда само токени с индекс $j <= i$.

#figure(
  image("../../images/Attention_diagram_bw_swapped.png", width: 78%),
  caption: [Attention механизъм и контекстно претегляне],
)

=== 4.2. Функционални модули и зависимости

Функционално решението е изградено от следните модули:

- *Data Module* - ingest, cleaning, dataset sharding, статистики, проверка на входните файлове.
- *Training Module* - optimizer loop, stages, gradient accumulation, checkpoint/recover.
- *Inference Module* - prefill/decode, sampling режими, KV cache, performance counters.
- *Evaluation Module* - benchmark сценарии и сравнение на ключови метрики.
- *Deployment Module* - Docker runtime, remote execution, data sync, artifact management.

Комуникацията между модулите е конфигурационно-ориентирана: модулите не разчитат на ръчно редактирани вътрешни параметри, а на explicit YAML settings, което намалява риска от скрити зависимости.

=== 4.3. Data и training pipeline

Data pipeline изпълнява нормализация, филтриране, токенизация и shard-ване, а получените shard файлове се подават директно към training loop-а за предвидим I/O.

Training pipeline е stage-based: всеки stage има собствени параметри (контекст, learning schedule, loss weights), а състоянието се пази с checkpoint, което позволява надежден resume при прекъсване.

Базовите training термини се дефинират така:

$ L_(A) = - frac(1, N) sum_(t=1)^N ln p(y_t | y_<t), quad L_(D) = - frac(1, N) sum_(t=1)^N ln p_(diff)(y_t | c_t) $

където $L_(A)$ е autoregressive cross-entropy термин, $L_(D)$ е diffusion/draft термин, $N$ е броят обучителни позиции в batch-а, $y_t$ е таргет токенът на позиция $t$, а $c_t$ е контекстът, достъпен за diffusion клона при съответната маска.

Общата loss комбинация в Anchor-TiDAR режима е:

$ L = alpha L_(A) + beta L_(D) + rho D_(f) + chi D_(r) + delta L_(H) + delta_(m) L_(P) + eta L_(S) + gamma L_(K) $

В практическата постановка се използват няколко вида loss термини: (1) базови езикови термини за AR и Diff клоновете, (2) дистрибуционни термини за подравняване между двата клона, (3) маскирани/префиксни термини за стабилност на acceptance веригата и (4) допълнителни регуляризационни термини за контрол на training динамиката.

$ D_(f) = sum_v p_(A)(v) ln frac(p_(A)(v), p_(D)(v)), quad D_(r) = sum_v p_(D)(v) ln frac(p_(D)(v), p_(A)(v)), quad L_(H) = - sum_t m_t ln p_(D)(y_t | c_t) $

#align(center)[
  #leg("3b bucket", bp.c-orange) #h(8pt)
  #leg("3b full", bp.c-red) #h(8pt)
  #leg("3b AR", bp.c-green) #h(10pt)
  #leg("KV full", kvp.c-red) #h(8pt)
  #leg("KV exact", kvp.c-blue) #h(8pt)
  #leg("KV bucket", kvp.c-orange)
]

#grid(
  columns: (1fr, 1fr),
  gutter: 6pt,
  [
    #bp.line-chart(
      "A40 GPU: 3b/k=16 (bucketed vs full AR)",
      (
        (label: "3b bucketed", color: bp.c-orange, data: bp.m3b_steady),
        (label: "3b full-context", color: bp.c-red, data: bp.m3b_full_steady),
        (label: "3b AR", color: bp.c-green, data: bp.m3b_ar_steady),
      ),
      width: 82mm,
      height: 35mm,
      y-label: "steady tokens/s",
    )
  ],
  [
    #kvp.line-chart(
      "KV policy: 3b first",
      kvp.m3b_first,
      width: 82mm,
      height: 35mm,
      y-label: "compile+first tokens/s",
      y-label-dx: 2pt,
    )
  ],
)

Двете графики показват как политиката за оразмеряване на KV cache влияе директно върху decode производителността: bucketed подходът запазва по-висока steady скорост спрямо full-context baseline, а в first-run режима се вижда цената на compile и ефектът от cache sizing върху първата заявка.

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 5pt,
  [#align(center)[#scale(88%)[#fts.line-chart("Kernel terms (7B-style, lower is better)", fts.k7b, width: 53mm, height: 34mm, y-label: "ms / attention")]]],
  [#align(center)[#scale(88%)[#fts.line-chart("Sampling-only 3b (lower is better)", fts.samp3b, width: 53mm, height: 34mm, y-label: "ms / sampling cycle")]]],
  [#align(center)[#scale(88%)[#fts.line-chart("MLP only (7B-style, lower is better)", fts.mlp7b, width: 53mm, height: 34mm, y-label: "ms / MLP")]]],
)

#align(center)[
  #leg("struct", rgb("#1D4ED8"), size: 7.4pt) #h(8pt)
  #leg("dense", rgb("#DC2626"), size: 7.4pt) #h(8pt)
  #leg("d+0", rgb("#059669"), size: 7.4pt) #h(8pt)
  #leg("AR", rgb("#7C3AED"), size: 7.4pt) #h(8pt)
  #leg("TiDAR", rgb("#0F766E"), size: 7.4pt) #h(8pt)
  #leg("top-k", rgb("#EAB308"), size: 7.4pt)
]

Тези 3 графики показват Free Token Slots феномена при attetnion, sampling и MLP.

#pagebreak()

#set par(justify: false)

#align(left)[#text(weight: "semibold")[SmolLM 135M vs 360M]]

#grid(
  columns: (auto, auto, auto),
  gutter: 10pt,
  [#box(width: 7pt, height: 7pt, fill: rg.color-135, radius: 1pt)[] #h(3pt) #text(size: 8pt)[135M]],
  [#box(width: 7pt, height: 7pt, fill: rg.color-360, radius: 1pt)[] #h(3pt) #text(size: 8pt)[360M]],
  [#box(width: 7pt, height: 7pt, fill: rg.color-delta, radius: 1pt)[] #h(3pt) #text(size: 8pt)[ratio 135/360]],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  [
    #rg.compare-chart(
      (
        (label: "135M", color: rg.color-135, data: rg.diff-135),
        (label: "360M", color: rg.color-360, data: rg.diff-360),
      ),
      width: 82mm,
      height: 34mm,
      y_label: "Diff loss",
      delta_data: rg.delta-ratio-series(rg.diff-135, rg.diff-360),
    )
  ],
  [
    #rg.compare-chart(
      (
        (label: "135M", color: rg.color-135, data: rg.acc-135),
        (label: "360M", color: rg.color-360, data: rg.acc-360),
      ),
      width: 82mm,
      height: 34mm,
      y_label: "Greedy acc",
      delta_data: rg.delta-ratio-series(rg.acc-135, rg.acc-360),
    )
  ],
)

По-големият 360M модел поддържа по-стабилен quality профил при сравним decode режим, което е очакван индикатор за по-висок моделeн капацитет.

В practical inference план това означава, че 360M дава по-предсказуемо поведение при сходни decode настройки, докато 135M остава полезен за по-евтини и бързи итерации. Тази двойка графики е полезна за избор на model size спрямо budget/quality изисквания.

#v(4pt)
#align(center)[
  #leg("Stable baseline", tsc.stable-color, size: 7.1pt) #h(7pt)
  #leg("KL-only no-diff", tsc.kl-only-color, size: 7.1pt) #h(7pt)
  #leg("KL-keep soft-align", tsc.kl-keep-color, size: 7.1pt) #h(7pt)
  #leg("Distill AR->Diff", tsc.distill-color, size: 7.1pt)
]
#align(center)[
  #leg("G-eta hard-agree", tsc.smallar-color, size: 7.1pt) #h(7pt)
  #leg("B-beta diff-boost", tsc.biggerbeta-color, size: 7.1pt) #h(7pt)
  #leg("Top-K set-match", tsc.topk-color, size: 7.1pt) #h(7pt)
  #leg("B-gamma hard+set", tsc.biggamma-color, size: 7.1pt) #h(7pt)
  #leg("D-mask prefix-only", tsc.maskedlater-color, size: 7.1pt)
]

Следващите графики показват delta спрямо stable baseline за core и за разширения набор от loss конфигурации върху TinyStories (90k-121k).

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  [
    #align(left)[#text(size: 8.5pt, weight: "semibold")[Core loss варианти]]
    #tsc.multi-line-chart(
      (
        (color: tsc.kl-only-color, data: tsc.delta-greedy-kl-only),
        (color: tsc.kl-keep-color, data: tsc.delta-greedy-kl-keep),
        (color: tsc.distill-color, data: tsc.delta-greedy-distill),
      ),
      width: 82mm,
      height: 34mm,
      y_label: "Delta Greedy",
      y_tick_percent: true,
      x_min: tsc.branch-x-min,
      x_max: tsc.branch-x-max,
      y_min: tsc.delta-greedy-y-min,
      y_max: tsc.delta-greedy-y-max,
    )
  ],
  [
    #align(left)[#text(size: 8.5pt, weight: "semibold")[Разширен loss набор]]
    #tsc.multi-line-chart(
      (
        (color: tsc.smallar-color, data: tsc.delta-greedy-smallar),
        (color: tsc.biggerbeta-color, data: tsc.delta-greedy-biggerbeta),
        (color: tsc.topk-color, data: tsc.delta-greedy-topk),
        (color: tsc.biggamma-color, data: tsc.delta-greedy-biggamma),
        (color: tsc.maskedlater-color, data: tsc.delta-greedy-maskedlater),
      ),
      width: 82mm,
      height: 34mm,
      y_label: "Delta Greedy",
      y_tick_percent: true,
      x_min: tsc.branch-x-min,
      x_max: tsc.branch-x-max,
      y_min: tsc.delta-greedy-ext-y-min,
      y_max: tsc.delta-greedy-ext-y-max,
    )
  ],
)

Анализ: TinyStories е нискоентропиен корпус с ограничена лексикална вариативност, затова разликите между loss вариантите често са с малка амплитуда. За по-ясно разделяне на ефектите е необходимо скалиране към по-големи модели и по-трудни корпуси.

Core вариантите са добър избор за стабилност и контрол, докато разширеният набор е по-подходящ за търсене на агресивни speedup хипотези.

#v(4pt)
#pagebreak()
#align(left)[#text(weight: "semibold")[Логитни хистограми (TinyStories примери)]]
За всяка позиция са показани два панела: горният е AR verify разпределението, а долният е Diff draft разпределението за същите токени в същия ред.
#v(2pt)
#scale(96%)[#tsc.render-hist(tsc.meta2, tsc.data2)]
#v(3pt)
#scale(96%)[#tsc.render-hist(tsc.meta3, tsc.data3)]
#v(2pt)
След първия reject често се наблюдават локални top-1 съвпадения между AR и Diff, но acceptance веригата остава прекъсната за текущата итерация, поради което позициите не се маркират като приети.

#pagebreak()

#set par(justify: true)

=== 4.4. Anchor-TiDAR: hybrid decode логика

#figure(
  image("../../images/Images_TiDAR_Optimization/Anchor_TiDAR_forward_pass.png", width: 100%),
  caption: [Anchor-TiDAR forward pass: verify + predraft в един моделeн проход],
)

Anchor-TiDAR въвежда sequence-level hybrid decode: системата генерира draft предложения паралелно и ги верифицира autoregressively в рамките на един forward pass със специализирани маски. Това позволява по-добро използване на GPU паралелизма спрямо класическия AR decode път.

Логически decode цикълът е:

1. prefill на контекста и инициализация на KV cache;
2. генериране на draft кандидати;
3. verify стъпка и приемане/отхвърляне на префикс;
4. commit/rollback на pointer-и в KV cache;
5. преход към следваща итерация.

#figure(
  image("../../images/TiDAR_decode_AND_attention_mask_and_agenda.png", width: 88%),
  caption: [TiDAR маски: decode и training с обща легенда],
)

Практическата полза е, че verify/predraft изчисленията се амортизират в един компактен execution path, което намалява относителния overhead при decode.

=== 4.5. Free Token Slots и GPU ефективност

Концепцията Free Token Slots описва случаи, при които в дадена decode итерация има неизползван паралелен ресурс, който може да бъде запълнен с полезна допълнителна работа. Този анализ е ключов за latency-critical deployment сценарии, защото показва къде архитектурните промени носят реална speedup стойност.

#figure(
  [
    #box(width: 168mm, height: 64mm)[
      #hh.bar-chart(
        "Speedup vs AR (steady TPS)",
        hh.overlay_bars,
        width: 168mm,
        height: 64mm,
      )
      #place(top + left, dx: 35mm, dy: 16pt)[
        #text(size: 8pt, weight: "semibold", fill: rgb("#374151"))[NVIDIA A40]
      ]
      #place(top + left, dx: 111mm, dy: 16pt)[
        #text(size: 8pt, weight: "semibold", fill: rgb("#374151"))[NVIDIA H100]
      ]
    ]
  ],
  caption: [Head-to-head speedup диаграма (steady TPS), рендерирана директно от Typst],
)

Валидираните резултати в този проект показват, че hybrid decode подходът е устойчив при мащаби до 7B параметри и може да осигури значимо ускорение спрямо AR baseline при запазване на конкурентно качество.

=== 4.6. End-to-end decode път (prefill + decode)

#figure(
  grid(
    columns: (50%, 50%),
    gutter: 8pt,
    align: top,
    [
      #image("../../images/TiDAR_prefill_mask.png", width: 100%)
      #v(4pt)
      #align(center)[#text(size: 10pt)[TiDAR prefill mask]]
    ],
    [
      #image("../../images/Prefill_and_Decode_diagram_wide.png", width: 100%)
      #v(4pt)
      #align(center)[#text(size: 10pt)[Prefill + decode path]]
    ],
  ),
  caption: [End-to-end decode път в GIANT/TiDAR: prefill маска (ляво) и decode pipeline (дясно)],
)

Практически inference изпълнението е разделено на две фази. В *prefill* фазата целият prompt се обработва наведнъж, като се инициализира KV cache състоянието за последващото генериране. В *decode* фазата системата добавя нови токени итеративно, като използва кешираната история вместо повторно пресмятане на целия контекст.

При GIANT този процес следва класически AR модел. При TiDAR/Anchor-TiDAR към същия execution skeleton се добавят verify/predraft стъпки и структурирани маски, без да се нарушава базовата консистентност на KV cache. Това позволява сравнение между режимите в една и съща инфраструктура и при едни и същи входни условия.

Поради тази унифицирана структура измерванията за скорост, закъснение и стабилност са директно съпоставими между AR и hybrid decode вариантите, което е критично за обективна оценка на реалния practically usable speedup.

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
  kind: image,
  caption: [KV cache commit/rollback при Anchor-TiDAR: поток и pointer update],
)

#pagebreak()

== 5. Реализация

Изборът на технологични средства е направен с фокус върху надеждност и ефективност:

- *Python* като основен език за бърза разработка и богата ML екосистема.
- *JAX/Flax/Optax* за висока числена производителност, модулност и устойчив checkpointing.
- *Docker + S3 workflow* за reproducible runtime, пренос и съхранение на големи данни.
- *Remote GPU execution* за тежки training/inference run-ове.

JAX стекът е особено силен в този проект, защото комбинира *JIT* компилация и *XLA* оптимизации с функционален модел на изчисление. На практика това позволява training/inference функциите да се компилират до оптимизиран изпълним граф за конкретни tensor shapes и така да се намали host overhead. Flax дава ясен модел за архитектура и параметри, Optax осигурява стабилна optimizer/scheduler логика, а Orbax гарантира надеждно checkpoint възстановяване при дълги run-ове.

Използваните алгоритми включват:

- decoder-only autoregressive training;
- speculative/hybrid decode стратегии;
- KV cache bucket policy;
- анализ на acceptance и локални logits профили;
- loss composition за TiDAR post-training sweep.

От инженерна гледна точка изборът на този стек е обоснован и с изискването за дългосрочна поддръжка. Конфигурациите, checkpoint файловете и benchmark артефактите са съвместими между локална и remote среда, което позволява един и същ експеримент да бъде повторен при идентични параметри и сравним хардуер.

Допълнително, modular entry-point структурата намалява риска от монолитни скриптове с неявни зависимости. Всеки сценарий (data, train, inference, evaluate) е отделен и може да бъде автоматизиран в CI/automation pipeline.

Използваната литература включва фундаменталните Transformer публикации, работи за serving оптимизации и референтната статия за TiDAR.

#pagebreak()

== 6. Описание на приложението

=== 6.1. Стартиране и инсталация

Препоръчителният начин за стартиране е чрез предоставения Docker контейнер (наличен автоматично на `docker.io/bonanc/giant-training:latest` или построен локално от CICD папката), но приложението може да се изпълнява и директно в Linux/macOS/WSL2 среда с Python. Типичният процес е:

1. клониране на репозиторито;
2. настройка на dependencies/venv или Docker контейнер;
3. проверка на конфигурационните файлове;
4. стартиране на data pipeline, training или inference entry point.

Примерни команди:

```bash
python GIANT/v2/data_pipeline/build_corpus.py --config OpenWebText_1b.yml
python GIANT/v2/model/Run_training.py --resume latest --batch 64
python GIANT/v2/model/Generate_faster.py --prompt "Steve Jobs is"
python TiDAR/model/Run_training.py --config TiDAR/model/Config_135m.yml
python TiDAR/model/inference.py --prompt "What is KV cache" --draft_len 8
```

=== 6.2. Използване

Потребителят работи изцяло през CLI, като управлява run параметрите чрез YAML конфигурации и command flags. Основните режими са:

- подготовка на данни;
- обучение (базов режим или TiDAR режим);
- inference (AR или Anchor-TiDAR);
- benchmarking и анализ на метрики чрез Typst.

=== 6.3. Поддръжка

Поддръжката е организирана около три практики: версия на конфигурациите, периодично архивиране на checkpoints/логове и проверка за съвместимост между model state и tokenizer. При промени в кода се изпълнява кратък benchmark run за регресии, за да се запази проследимостта на резултатите.

== 7. Заключение

Проектът GIANT + Anchor-TiDAR изпълнява основната си цел: реализирана е цялостна и работеща платформа за езикови модели, която покрива всички ключови етапи - подготовка на данни, обучение, инференс и измерване на производителността. Всичко това е обединено в една обща codebase с възпроизводим процес на изпълнение. Системата може да се използва както за практически deployment сценарии, така и за контролирани изследователски експерименти.

Ключовият принос е архитектурното разделяне между стабилен core слой (GIANT) и изследователски extension слой (TiDAR). Това позволява паралелно развитие на reliable training/inference pipeline и на нови decode идеи, без нарушаване на базовата експлоатация. В operational план платформата поддържа checkpoint/resume, ясни CLI entry points, KV cache оптимизации и проследими експериментални артефакти.

Емпиричните резултати показват, че hybrid decode подходът (draft + autoregressive verify в един forward pass) може да съчетае висока скорост и запазено качество. В проведените head-to-head измервания върху модели до 7B параметри е отчетено ускорение от 4.93x до 9.75x спрямо AR baseline при стабилно inference поведение в различни GPU режими. Ограничението в текущата версия остава loss sweep експериментът, който трябва да се разшири към по-голям model scale, за да се разграничат по-ясно фините ефекти между отделните loss комбинации.

\

В близкото бъдеще развитието на проекта е насочено към:

- multi-GPU distributed training и по-големи тренировъчни диапазони;
- разширяване и скалиране на базовия модел към диапазон 300M-1B параметъра;
- обучение върху по-богати и по-разнообразни корпуси (OpenWebText, Wikipedia, OpenCrawl и сходни източници) за по-добра обобщаемост;
- доразвитие на serving слоя за по-висок паралелизъм на заявки и по-ефективно KV cache управление;
- експерименти с архитектурни вариации за decode speedup (MoE, MLA и сходни подходи) и проследяване на ефекта им върху Free Token Slots.
