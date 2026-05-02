# Teacher output-KD

This is the first distillation path for GIANT v3.

It has two offline steps:

- [build_prompt_bank.py](build_prompt_bank.py) extracts user-turn prompt prefixes from chat/SFT datasets
- [teacher_distill.py](teacher_distill.py) runs a HuggingFace teacher model and writes normal chat SFT rows

The output is intentionally just another chat dataset with `messages`. The existing data pipeline can consume it with assistant-only loss masks.

## Why no logprob KD yet

Most current frontier GIANT runs use custom student tokenizers. A HF teacher like Qwen/Llama/Mistral has a different tokenizer, so teacher token logprobs do not align with student tokens.

That makes direct logprob KD wrong unless we add a token-alignment layer or restrict KD to teacher/student pairs with the same tokenizer.

So v1 is output-KD:

- prompt bank row ends with a user message
- teacher generates an assistant message
- student trains on the assistant message only

Future expansion:

- same-tokenizer logprob KD
- candidate-answer reranking KD
- text-span/token alignment for cross-tokenizer KD

## Example config

Use [configs/teacher_kd_smoltalk_example.yml](configs/teacher_kd_smoltalk_example.yml) as the starting point.
