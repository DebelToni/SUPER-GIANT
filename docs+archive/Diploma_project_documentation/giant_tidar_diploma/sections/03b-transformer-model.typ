=== 2.5 Основен математически модел на трансформера

==== 2.5.1 Теоретична отправна точка: Attention Is All You Need

Теоретичната основа на системата е архитектурата Transformer, въведена от Ashish Vaswani и съавтори в статията "Attention Is All You Need" (2017). Ключовата идея е да се замени рекурентната обработка с attention механизъм, който позволява паралелна обработка на целия токенен контекст. Това е фундаментално за LLM, защото training и inference трябва да използват максимално добре паралелизма на GPU.

В оригиналния формализъм attention операторът е:

$ A(Q, K, V) = S( frac(Q K^T, sqrt(d_k)) + M ) V $

където `Q`, `K`, `V` са query/key/value матрици, `M` е маска, а `S` е softmax функцията.

$ S(z_i) = frac(exp(z_i), sum_j exp(z_j)) $

Формулировката от оригиналния текст (както е изписана в paper-а, без експлицитно показана маска) е:

$ op("Attention")(Q, K, V) = op("softmax")( frac(Q K^T, sqrt(d_k)) ) V $

Multi-head формата (както е на фигурата) е:

$ op("MultiHead")(Q, K, V) = op("Concat")(op("head")_1, dots.c, op("head")_h) W^O $

където:

$ op("head")_i = op("Attention")(Q W_i^Q, K W_i^K, V W_i^V) $

Еквивалентно, в компактна форма:

$ H(Q, K, V) = [h_1 ‖ dots.c ‖ h_n] W^O, h_i = A(Q W_i^Q, K W_i^K, V W_i^V) $

#figure(
  image("../../../images/mha_img_original.png", width: 68%),
  caption: [Multi-head attention: блокова схема и базови формули],
)

==== 2.5.2 От класически Transformer към decoder-only LLM

Оригиналната работа описва encoder-decoder архитектура. В GIANT се използва decoder-only вариант, подходящ за next-token prediction. Практически това означава:

- причинно (causal) маскиране в self-attention;
- autoregressive objective за следващ токен;
- изцяло еднопосочен токенен поток при генерация.

Един блок в decoder-only варианта може да се запише като:

$
x' = x + H(N(x), N(x), N(x))
$

$
y = x' + F(N(x'))
$

където `N` е нормализация, а `F` е feed-forward модул.

#grid(
  columns: (26%, 74%),
  gutter: 10pt,
  align: top,
  [
    #image("../../../images/DecoderTransformer.png", width: 100%)
  ],
  [
    *Фигура: Decoder Transformer архитектура*

    Диаграмата показва базовия поток в decoder-only модел: входни токени -> embedding -> последователност от Transformer блокове -> нормализация -> проекция към логити. В GIANT тази структура се реализира с модерни компоненти като RoPE, RMSNorm и SwiGLU, а при инференс се допълва с KV cache за ускорение.


    Изборът на тези "модерни попълнения" е направен като естествена еволюция спрямо оригиналния Transformer от 2017. В *Attention Is All You Need* се използват LayerNorm (а не BatchNorm), ReLU-базиран feed-forward блок и синусоидални позиционни вектори. В настоящата реализация са предпочетени RMSNorm (по-лек и стабилен при decoder-only LLM), SwiGLU (по-добра експресивност и практическа ефективност спрямо базовите активации) и RoPE (по-устойчива позиционна индукция при по-дълги контексти). Така архитектурата запазва теоретичната основа от 2017, но е адаптирана към съвременния LLM performance профил и в практиката е по-близка до модерен LLaMA-style decoder stack.
  ],
)

==== 2.5.3 Нормализация, активации и позиционна информация в GIANT

Вместо класически LayerNorm, в модела се използва RMSNorm, което е по-леко и работи стабилно в големи decoder-only конфигурации; в този смисъл архитектурата на GIANT е по-близка до съвременен LLaMA-подобен дизайн, отколкото до оригиналния Transformer от 2017.

$
R(x) = frac(x, sqrt(frac(1, d) sum_(i=1)^d x_i^2 + epsilon)) * g
$

Feed-forward частта използва SwiGLU тип гейтинг:

$
G(x) = s(x W_u) * (x W_v), quad s(t) = t sigma(t)
$

Позиционната информация се подава чрез RoPE, което позволява по-добро поведение при по-дълги контексти и е стандартен избор в съвременните decoder-only LLM реализации.

==== 2.5.4 Обучителна цел (AR) и връзка с TiDAR

Базовата objective функция за GIANT е autoregressive cross-entropy:

$
L_1 = - frac(1, N) sum_(b,t) m_(b,t) log p_theta(x_(b,t+1) | x_(b, <= t))
$

където `L_1` е AR (next-token) cross-entropy.

Примерен изход от GIANT v2 checkpoint, трениран върху корпус със силно Wikipedia присъствие (заедно с други източници), е:
```text
User: What is the capital of France?
Assistant: The capital of France is sometimes called Paris.<EOS>
```

- размер на модела: приблизително 101 милиона параметъра;
- обем на обучението: приблизително 2 милиарда токена.

За мащабно сравнение:
- Llama 3 70B - trained on ~15T tokens;
- DeepSeek R1 671B - trained on ~14.8T tokens.

TiDAR надгражда този математически модел, като преизползва същата Transformer основа и добавя diffusion-режим на внимание, специални маски и допълнителни loss термини за съгласуване между AR и Diff логити.
