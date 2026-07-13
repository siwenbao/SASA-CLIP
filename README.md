# SASA-CLIP: Structure-Aware Alignment with a Gaussian Prior for Fine-Grained Video Action Recognition

This repository provides the LLM-generated descriptors and the descriptor
generation pipeline used in our paper, for reproducibility.

## Contents

- `prompts/` — the system prompt and user prompt template used to generate descriptors.
- `descriptors/` — the generated spatial and temporal descriptors for all Kinetics-400 categories.
- `scripts/` — the descriptor generation script.

## Descriptor Generation

Descriptors are generated with the OpenAI GPT-4.1 API.

| Setting | Value |
|---|---|
| Model | GPT-4.1 |
| Temperature | 0.7 |
| Temporal descriptors per class | 3 (start / middle / end) |
| Spatial descriptors per class | 3 |

For each action category, the model produces:
- **Temporal descriptors**: three descriptors following the stage order
  *start → middle → end*, each describing a visually distinct phase with an
  explicit spatial anchor (body part, scene element, or prop).
- **Spatial descriptors**: three single-frame static cues describing the most
  distinctive, class-identifying visual elements.

Each descriptor is at most 15 words.

## Post-generation Validation

Every generated set is automatically validated; if any rule fails, generation
is retried (up to 6 attempts):

- each descriptor ≤ 15 words;
- each temporal descriptor contains an explicit spatial anchor;
- spatial descriptors contain no temporal-order words (then, after, while, ...);
- no vague words (some, maybe, ...) or mental-state words (think, want, feel, ...);
- all descriptors within a class are unique;
- temporal descriptors follow the exact order start, middle, end.

## Usage

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python scripts/generate_descriptors.py \
    --csv-path labels/kinetics_400_labels.csv \
    --model-name gpt-4.1 \
    --n-temporal 3 --n-spatial 3 --temperature 0.7
```

## Citation

If you find this useful, please cite our paper (details to be added upon publication).
