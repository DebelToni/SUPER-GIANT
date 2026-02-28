<h1 align="center">SUPER GIANT</h1>
<!-- <h5 align="center"> SUPERsupreme Utra PROfesional ELITE ReVolutIonary GIGA intelligent ArtIfical neXus TITAN </h5> -->

GIANT is my own custom implementation of a large language model (LLM) written in Python - JAX.
It is designed to be a modern, robust and easily expandable implementation with a focus on performance on a single GPU and ease of use. 

This project is developed for my own learning but also serves as my highschool graduation project. You can find the full bulgarian documentation for that here:
[Bulgarian Graduation Project Documentation](<docs/Diploma_project_documentation/giant_tidar_diploma/Дипломна Работа Антон Христов.pdf>)

To understand how LLMs work watch my first video:
<br>

[![Watch on YouTube — ZaBJ2VwDvPI](https://img.youtube.com/vi/ZaBJ2VwDvPI/maxresdefault.jpg)](https://youtu.be/ZaBJ2VwDvPI)
The original video showcased the [v0](GIANT/v0) implementation and some performance speeds from the [v1](GIANT/v1) implementation which you can find under [GIANT](GIANT/).

Newest version of [GIANT is v2](GIANT/v2) which focuses on scaling the model and the data, bigger and better [data pipeline](GIANT/v2/data_pipeline/) and ChatBot behavior. Example responce from GIANTv2:
```
User: What is the capital of France?
Assitant: The capital of France is sometimes called Paris.<EOS>
```
> (This is a 101 milion param checkpoint trained on 2 bilion tokens including a schedule and a circulumn of basetext, wikipedia, webtext and finally chat examples.)

## What I am working on right now:
- Right now I am working on replicating the results of the [TiDAR paper](https://arxiv.org/pdf/2511.08923) under my [TiDAR](TiDAR/) folder.
- I am using GIANT as the underlying LLM architecture and data pipeline. Only the training script differs significantly because TiDAR is a special diffusion based post-training method. 
- I've implemented a small architectural change to original TiDAR proposed in the paper which boosts drafting efficiency per step by a small amount. [>Details<]()
- I am also experimenting with different Loss functions for TiDAR. [>Details<]()

* When finished I want to experiment with the TiDAR "free token slots" fenomenon when paired with memory optimizaion architectural changes like [MLA], [MoE], [Sliding window attention] and [Router aware drafting]


## Architecture of GIANT
- Classic Decoder-only transformer architecture
- Modern modules used such as `RMSnorm`, `SwiGLU`, `RoPE`
- Training supports batching, training on custom private data or online Teacher distillation
- Powered by 𝗝𝗔𝗫’𝘀 𝗝𝗜𝗧 𝗰𝗼𝗺𝗽𝗶𝗹𝗮𝘁𝗶𝗼𝗻, 𝗰𝘂𝗗𝗡𝗡’𝘀 𝗙𝗹𝗮𝘀𝗵 𝗔𝘁𝘁𝗲𝗻𝘁𝗶𝗼𝗻 𝗸𝗲𝗿𝗻𝗲𝗹𝘀, 𝗮𝗻𝗱 𝗮 𝗞𝗩 𝗰𝗮𝗰𝗵𝗲 for faster inference

> [!NOTE]
> Future Architectural features:
> - Mixture of Experts (MoE)
> - Multi-head Latent Attention (MLA)
> - Real time access of tools at inference time (in the TTC) - see [TRM as tool use](TRM/TRM-token-tool)
> - Diffusion future prediction for faster text generation - currently implementing this in the [TiDAR](TiDAR/]) folder
> 

---

## Installation

I've created a base Docker image with all the dependencies installed for easy use - `bonanc/giant-training:latest`

(this is the recommended version with no constant sync S3, for constant Minio sync see [here](CICD/Docker/giant-training-S3/) )

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
