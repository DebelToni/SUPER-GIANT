
<h1 align="center">SUPER GIANT</h1>
<p align="center"> SUPERsupreme Utra PROfesional ELITE ReVolutIonary GIGA intelligent ArtIfical neXus TITAN </p>

GIANT is my custom implementation of a large language model (LLM) written in Python - JAX.
It is designed to be efficient, scalable and robust, with a focus on performance on a single GPU and ease of use.
<br>

[![Watch on YouTube — ZaBJ2VwDvPI](https://img.youtube.com/vi/ZaBJ2VwDvPI/maxresdefault.jpg)](https://youtu.be/ZaBJ2VwDvPI)

<br>
Right now it is in an active state of development with working on v1 - the 'second' version of the model which greatly improves v0 in terms of performance and flexability.
<br><br>
## Architecture
- Classic Decoder-only transformer architecture
- 𝗚𝗜𝗔𝗡𝗧 𝘃𝟬 -> 𝗚𝗜𝗔𝗡𝗧 𝘃𝟭 — standard Transformer decoder-only LLMs
- Powered by 𝗝𝗔𝗫’𝘀 𝗝𝗜𝗧 𝗰𝗼𝗺𝗽𝗶𝗹𝗮𝘁𝗶𝗼𝗻, 𝗰𝘂𝗗𝗡𝗡’𝘀 𝗙𝗹𝗮𝘀𝗵 𝗔𝘁𝘁𝗲𝗻𝘁𝗶𝗼𝗻 𝗸𝗲𝗿𝗻𝗲𝗹𝘀, 𝗮𝗻𝗱 𝗮 𝗞𝗩 𝗰𝗮𝗰𝗵𝗲 for faster inference

> [!NOTE]
> Future Architectural Ides:
> - Mixture of Experts (MoE) layers
> - Real time access of tools at inference time (in the TTC)
> - Diffusion layers for thought and text generation
> - Multi-frequency layers (my idea)

---

Level of V0 intelligence 0.1%:
```bash
Original prompt: One sunny day,

==================== RESULT ====================
One sunny day, Lily went to the park with her mom. She saw a big hill and wanted to see
 what was inside. She ran to the hill and went up around the hill with her hands. She saw a
big hole. She was scared, but she was too high. She saw many flowers on the ground<EOF>
```
V0 was trained on 10% of the TinyStories dataset for 10 minutes and alwready could form semi-coherent simple sentances.


V1 features:
- Full-dimensional Key-Value cache for faster inference
- JAX-based Flash attention for overall performance
- Dynamic allocation and parsing of the database 
- Overall performance and architecture improvements
(still in development - features are implemented but more training tests are needed)

Level of v1 intelligence 0.100001#: 
- not improved because still using the same dataset, but speed and performance is increased
```bash
Little Timmy. He was only three years old. He was very curious and wanted to explore the world.

One day, he decided to go on an adventure. He put on his shoes and grabbed his bag. He walked and walked until he came to a big tree. He looked up and saw a big, beautiful tree. He wanted to climb it, but he was too scared.

He started to climb the tree. He was so excited! He climbed up the tree and looked around.

tokens_per_second_decode: 89.301928
GPU: NVIDIA GeForce RTX 3060 
```


V2 goals:
- ChatBOT training data and knowledge base

> Weights for more capable future models will be available on huggingface

---

## Model architecture card
( a bit outdated )

[Read the model overview](Model_Overview.md)

