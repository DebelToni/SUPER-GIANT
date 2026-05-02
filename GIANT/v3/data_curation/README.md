# v3 data curation

This folder holds small scripts that create higher-signal training data around the raw corpora. It is not the main tokenizer/training pipeline; it prepares inputs that [../data_pipeline/](../data_pipeline/) later packs into Arrow shards.

## Current scripts

- [curated_wikipedia_search.py](curated_wikipedia_search.py) - stream/search English Wikipedia for target-aware factual passages
- [export_top_candidates.py](export_top_candidates.py) - export top ranked curated rows per target
- [build_demo_target_pack.py](build_demo_target_pack.py) - build human-readable demo target packs
- [translate_curated_booster.py](translate_curated_booster.py) - translate curated English booster rows into Bulgarian and write a bilingual JSONL
- [translate_smoltalk_to_bg.py](translate_smoltalk_to_bg.py) - translate selected SmolTalk conversations into Bulgarian and write original/translated/bilingual JSONL files
- [build_prompt_bank.py](build_prompt_bank.py) - extract user-turn prompt prefixes from chat/SFT datasets
- [teacher_distill.py](teacher_distill.py) - ask a HF teacher model to answer prompt-bank rows and write output-KD SFT JSONL
- [quality_filter/](quality_filter/) - Bulgarian quality-filter training and corpus filtering

## Chat-stack data examples

- curated English booster data config: [../Configs/Data/giant_chat_curated_booster_strong56.yml](../Configs/Data/giant_chat_curated_booster_strong56.yml)
- bilingual booster data config: [../Configs/Data/giant_chat_curated_booster_bg_en_bpe32k.yml](../Configs/Data/giant_chat_curated_booster_bg_en_bpe32k.yml)
- translated SmolTalk SFT data config: [../Configs/Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml](../Configs/Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml)
- teacher-KD prompt/data config example: [configs/teacher_kd_smoltalk_example.yml](configs/teacher_kd_smoltalk_example.yml)

## Data layout used by current chat experiments

- curated booster source: `/proj/giant-data/GIANT/GIANT-Chat/data_curation/strong56_wikipedia_v2/`
- bilingual booster source: `/proj/giant-data/GIANT/GIANT-Chat/data_curation/curated_booster_bg_en_v1/`
- translated SmolTalk source: `/proj/giant-data/GIANT/GIANT-Chat/data_curation/smoltalk_bg_en_v1/`
- packed datasets are not written here; they are written by [../data_pipeline/build_corpus.py](../data_pipeline/build_corpus.py)

## Notes

- Translation scripts need `torch`, `transformers`, and `sentencepiece`. The main JAX env may not have those.
- `translate_smoltalk_to_bg.py --resume` resumes from its `state.json`.
- `teacher_distill.py` is output-KD/SFT first. Logprob KD is future work because current frontier runs usually use custom student tokenizers that do not match the teacher tokenizer.
- The current Marian translation path is good enough for lexical Bulgarian bootstrapping but not enough for real reasoning quality by itself.
- Always inspect samples before scaling translated SFT.
