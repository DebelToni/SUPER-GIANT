<h1 align="center">SUPER GIANT</h1>
<!-- <p align="center"> SUPERsupreme Utra PROfesional ELITE ReVolutIonary GIGA intelligent ArtIfical neXus TITAN </p> -->

GIANT is my custom implementation of a large language model (LLM) written in Python - JAX.
It is designed to be efficient, scalable and robust, with a focus on performance on a single GPU and ease of use.
<br>
<br>
Right now it is in an active state of development with working on v1 - the 'second' version of the model which greatly improves v0 in terms of performance and flexability.
<br><br>
## Architecture
- Classic Decoder-only transformer architecture
> [!NOTE]
> Future Architectural Ides:
> - Mixture of Experts (MoE) layers
> - My custom idea of 'Cross-contextual' attention
> - Real time access of tools at inference time (in the TTC)

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

V2 goals:
- ChatBOT training data and knowledge base


> Weights for more capable future models will be available on huggingface
