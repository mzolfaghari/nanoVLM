# nanoVLM Experiments

## Run: `nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_1xGPU_bs16_40000_lr_vision_5e-05-language_5e-05-0.00512_0515-154241`

### Training dataset

| Field | Value |
|-------|--------|
| **Name** | FineVision document presample (FVDoc v1) |
| **Path** | `/home/khashmi/data/qvac-vlm/finevision_doc_filter/presampled_dataset/fvdoc_v1_45-30-20-5_2026-05-12_imgcast` |
| **Format** | Hugging Face `datasets` on disk (569 Arrow shards, images embedded) |
| **Sampling** | Presampled mix **45-30-20-5** (document-filter pipeline; dated 2026-05-12, image-cast export) |
| **Schema** | Multi-image VQA/chat: `images`, `texts` (`user` / `assistant`), quality ratings (`relevance_*`, `image_correspondence_*`, `visual_dependency_*`, `formatting_*`) |
| **Loader** | `load_from_disk` → `VQADataset` (nanoVLM) |

### Model & training (summary)

| Field | Value |
|-------|--------|
| **Vision** | `google/siglip2-base-patch16-512` |
| **Language** | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| **Max image side** | 2048 |
| **LM context** | 4096 |
| **Hardware** | 1× GPU |
| **Effective batch** | 16 (`batch_size=2`, `gradient_accumulation_steps=8`) |
| **Steps (target)** | 40,000 |
| **Checkpoint evaluated** | `checkpoints/.../step_21500` (global step **21,500**) |
| **Learning rates** | vision `5e-5`, language `5e-5`, MP `0.00512` |

### Evaluation setup

| Field | Value |
|-------|--------|
| **Framework** | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) via `run_evaluation.py` / `eval.slurm` |
| **Tasks** | `mmstar`, `mmmu_val`, `ocrbench`, `textvqa_val`, `docvqa_val`, `scienceqa`, `mme`, `infovqa_val`, `chartqa` |
| **Merged results** | `eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_1xGPU_bs16_40000_lr_vision_5e-05-language_5e-05-0.00512_0515-154241/step_21500.json` |

### Evaluation results (step 21,500)

Primary metrics from merged lmms-eval results. Higher is better unless noted.

| Benchmark | Metric | Score |
|-----------|--------|------:|
| **MMStar** | Average | 0.278 |
| **MMMU (val)** | Accuracy | 0.248 |
| **OCRBench** | Accuracy | 0.359 |
| **TextVQA (val)** | Exact match | 0.267 |
| **DocVQA (val)** | ANLS | 0.530 |
| **ScienceQA** | Exact match | 0.206 |
| **MME** | Perception score | 736.5 |
| **MME** | Cognition score | 184.6 |
| **InfoVQA (val)** | ANLS | 0.183 |
| **ChartQA** | Relaxed overall | 0.534 |

#### MMStar breakdown

| Category | Score |
|----------|------:|
| Coarse perception | 0.356 |
| Fine-grained perception | 0.214 |
| Instance reasoning | 0.271 |
| Logical reasoning | 0.262 |
| Math | 0.310 |
| Science & technology | 0.256 |
| **Average** | **0.278** |

#### ChartQA breakdown

| Split | Relaxed accuracy |
|-------|-----------------:|
| Overall | 0.534 |
| Human split | 0.319 |
| Augmented split | 0.749 |

---

*Results generated from `step_21500.json` (global_step: 21500). MME reports sum-style subscores as returned by lmms-eval.*
