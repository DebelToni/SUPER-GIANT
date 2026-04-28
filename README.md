<a href="https://debeltoni.github.io/SUPER-GIANT" style="text-decoration: none; color: inherit;">
  <h1 align="center">SUPER GIANT</h1>
</a>
<!-- <h5 align="center"> SUPERsupreme Utra PROfesional ELITE ReVolutIonary GIGA intelligent ArtIfical neXus TITAN </h5> -->

## Overview

**SUPER-GIANT** is a complete framework for running **data preparation, training, and inference of custom Large Language Models**. It is the perfect starting point for new researchers that need a strong implementation to start from. It provides an end-to-end pipeline including:

- **Data preparation pipeline** - Integrates directly with HuggingFace datasets and tokenizers. We also support custom or imported fastText models for data curation and filtering.

- **Training loop** - Fully resumable and syncronized with S3 buckets, features multi-GPU training support (starting from [v3](GIANT/v3/)), which currently uses **Data Parallel (DP)** training, model sharding support planned for future releases

- **Inference pipeline** - fast decoding, KV-cache support, chat-style interactions

- **Speculative decoding** - highly optimized **Anchor-TiDAR** algorithm. Requires post-training modifications but achieves **5×–10× inference speedups** on GPUs

---

GIANT is the underlying LLM architecture for the LLM. Each version (v0, v1, v2, v3) builds with more modern changes to the original transfromer. It is my own version written in JAX but it is highly adjustable even only from YML configuration scripts.

It is designed to be a modern, robust and easily expandable implementation with a focus on performance on a single GPU and ease of use. 

This project is developed for my own learning but also serves as my highschool graduation project. You can find the full bulgarian documentation for that here:
[Bulgarian Graduation Project Documentation](<docs/Diploma_project_documentation/giant_tidar_diploma/Дипломна Работа Антон Христов.pdf>)

To understand how LLMs work watch my first video:
<br>

[![Watch on YouTube — ZaBJ2VwDvPI](https://img.youtube.com/vi/ZaBJ2VwDvPI/maxresdefault.jpg)](https://youtu.be/ZaBJ2VwDvPI)
The original video showcased the [v0](GIANT/v0) implementation and some performance speeds from the [v1](GIANT/v1) implementation which you can find under [GIANT](GIANT/).

## Architecture of GIANT v2
- Classic Decoder-only transformer architecture
- Modern modules used such as `RMSnorm`, `SwiGLU`, `RoPE`
- We also have full support for Encoders and also experimental modules such as XSA (Exclusive Self Attention)
- Training supports batching, training on custom private data or online Teacher distillation
- Powered by 𝗝𝗔𝗫’𝘀 𝗝𝗜𝗧 𝗰𝗼𝗺𝗽𝗶𝗹𝗮𝘁𝗶𝗼𝗻, 𝗰𝘂𝗗𝗡𝗡’𝘀 𝗙𝗹𝗮𝘀𝗵 𝗔𝘁𝘁𝗲𝗻𝘁𝗶𝗼𝗻 𝗸𝗲𝗿𝗻𝗲𝗹𝘀, 𝗮𝗻𝗱 𝗮 𝗞𝗩 𝗰𝗮𝗰𝗵𝗲 for faster inference
Current stable version of [GIANT is v2](GIANT/v2) which focuses on scaling the model and the data, bigger and better [data pipeline](GIANT/v2/data_pipeline/) and ChatBot behavior. Example responce from GIANTv2:
```
User: What is the capital of France?
Assitant: The capital of France is sometimes called Paris.<EOS>
```
> (This is a 38 milion param checkpoint trained on 500 milion tokens including a schedule and a circulumn of basetext, wikipedia, webtext and finally chat examples.)

## [Think in Diffusion, talk in AutoRegression](https://arxiv.org/pdf/2511.08923)
- NVIDIA's proposed model [TiDAR](TiDAR/) is already implemented (including my Anchor-TiDAR variant), and this is where I run ongoing experiments. It builds on top of the **GIANTv2** architecture
- I am currently validating/replicating the TiDAR paper behavior and refining training/inference settings on top of GIANT's core architecture and data pipeline. Results from my free token slots experiements can be found at [TiDAR/Docs/](TiDAR/Docs/)
- I've implemented a small architectural change to the original TiDAR design that improves drafting efficiency per step (always contirbuting atleast 1 token, no overhead). [>Details<](TiDAR/model/Optimize_TiDAR_worst_case_decoding.md)
- I am also experimenting with different loss function configurations for TiDAR. [>Details<](TiDAR/Docs/TinyStories_TiDAR_losses_Comparison.pdf)

> [!NOTE]
> Future Architectural features that will be test on top of GIANT and then Anchor-TiDAR:
> - Mixture of Experts (MoE)
> - Multi-head Latent Attention (MLA)
> - Sliding window attention and linear attetnion style mechanisms. The idea is to see how it affects the latency and performance of "free token slots".
> - On/Off Encoder mechanism - my own creation that combines TRM style processing by doing TiDAR style masking for swithing between the 2 modes in the same transformer. I want to see how this proves long [context retrieval](GIANT/v3/Long/). (could also say reasoning but my training budget is far too small to get to actual qualitative reasoning results).

## [GIANT v3](GIANT/v3/)
GIANT v3 is the current version I am working on, it features:
- Multi-GPU support with gradient accomultaion and sharding.
- DeepSeek style Mutlhead Latent Attention
- Stronger data curation

---

## Installation

I've created a base Docker image with all the dependencies installed for easy use - `bonanc/giant-training:latest`

(this is the recommended version with no constant sync S3, for constant Minio sync see [here](CICD/Docker/giant-training-S3/))
(i also have some [docs for coding agents](docs/AGENTS-documentation/) if you want to "vibe" out the training process)

You can run it with:
```bash
export TS_AUTHKEY="PUT YOUR TAILSCALE KEY HERE"
# if you rent compute in the cloud and want to access the container from anywhere. I recommend using RunPod.io if you don't have a GPU like me and want super easy and cheap GPU containers (I am not sponsored but very well could have been :) )

docker run --pull always -d --gpus all \
--name giant-training \
--mount type=bind,source="$HOME/GIANT",target=/proj \ # persistent storage path !
bonanc/giant-training:latest

docker exec -it giant-training bash
```
For a full rundown of `TS_*`/`SYNC_*` flags, tailnet configuration, and S3 sync behavior, see [CICD/Docker/README.md](CICD/Docker/README.md).
For easy use of S3 buckets use my amazing [s5cmd wrapper](CICD/tools/)
