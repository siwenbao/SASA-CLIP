"""
Descriptor generation for SASA-CLIP.

Generates ordered temporal and spatial descriptors for each Kinetics-400
action category using the OpenAI GPT-4.1 API.

Default settings (as reported in the paper):
    Model: GPT-4.1
    Temperature: 0.7
    Temporal descriptors per class: 3 (start / middle / end)
    Spatial descriptors per class: 3

Usage:
    export OPENAI_API_KEY=your_key
    python generate_descriptors.py \
        --csv-path labels/kinetics_400_labels.csv \
        --out-json descriptors/k400_descriptors.json \
        --out-meta-json descriptors/k400_descriptors_meta.json
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT_FULL = """You are a descriptor generator for video action recognition on Kinetics-400.

You must produce two descriptor groups per class:
1) Temporal descriptors (dynamic process semantics)
2) Spatial descriptors (single-frame static cues)

Global objective:
- Temporal descriptors are the core for fine-grained temporal reasoning.
- Spatial descriptors are visual anchors for frame-level alignment.
- Keep each descriptor concise and CLIP-friendly: <= 15 words.
- Avoid unverifiable, subjective, or mental-state statements.

Spatial descriptor rules (strict):
- Focus on the most distinctive, class-identifying visual elements.
- Can be a short noun phrase (e.g., "pull-up bar in a gym") or a brief
  scene description with key objects and spatial context.
- Must describe static, directly observable cues from a single frame.
- Do NOT use temporal-order words/process connectors, including:
  then, after, while, finally, next, before, during, later, once.
- Avoid vague words: some, kind of, sort of, maybe, etc.

Temporal descriptor rules (strict):
- Template: [action stage] + [key movement] + [spatial anchor].
- Stage token must come from the provided stage vocabulary and exact order.
- Key movement must be high-level and visually observable.
- Each temporal descriptor must include a concrete spatial anchor:
  a body part (arms/hands/legs/feet/...), a scene element
  (floor/ground/wall/air/...), or a prop (ball/bar/bow/target/...).
  A preposition cue (with/on/in/at/near/by/toward/in front of/...) is preferred
  but not strictly required if the anchor noun is present.
- Each temporal descriptor must capture a visually DISTINCT phase of the action.
  Avoid repeating the same body movement across stages.

Output format:
- Return ONLY valid JSON.
- No markdown, no explanations.
"""

SYSTEM_PROMPT_SPATIAL_ONLY = """You are a descriptor generator for video action recognition on Kinetics-400.

You must produce spatial descriptors (single-frame static visual cues) for each class.

Global objective:
- Spatial descriptors are visual anchors for frame-level alignment with CLIP.
- Keep each descriptor concise and CLIP-friendly: <= 15 words.
- Avoid unverifiable, subjective, or mental-state statements.
- Each descriptor should capture a DISTINCT visual element — do NOT repeat.

Spatial descriptor rules (strict):
- Focus on the most distinctive, class-identifying visual elements.
- Can be a short noun phrase (e.g., "pull-up bar in a gym") or a brief
  scene description with key objects, tools, or spatial context.
- Must describe static, directly observable cues from a single frame.
- Do NOT use temporal-order words/process connectors, including:
  then, after, while, finally, next, before, during, later, once.
- Avoid vague words: some, kind of, sort of, maybe, etc.

Output format:
- Return ONLY valid JSON.
- No markdown, no explanations.
"""

TEMPORAL_WORD_BLACKLIST = {
    "then", "after", "while", "finally", "next", "before", "during", "later", "once"
}
VAGUE_WORD_BLACKLIST = {
    "some", "maybe", "perhaps", "thing", "stuff"
}
MENTAL_STATE_BLACKLIST = {
    "think", "decide", "want", "feel", "believe", "plan", "intend", "hope"
}
ANCHOR_CUES = {
    "in", "on", "at", "near", "by", "with", "inside", "outside",
    "behind", "beside", "against", "above", "below", "under", "over",
    "toward", "towards", "around", "onto", "into", "across", "along",
    "through", "from", "off", "between", "amid",
}
# Body-part / environmental nouns that count as valid visual anchors
# even without a preposition (e.g., "raise arms rhythmically", "release arrow").
ANCHOR_NOUNS = {
    # body parts
    "arm", "arms", "hand", "hands", "leg", "legs", "foot", "feet",
    "finger", "fingers", "head", "body", "torso", "shoulder", "shoulders",
    "knee", "knees", "elbow", "elbows", "hip", "hips", "chest", "back",
    "face", "mouth", "eyes", "wrist", "wrists", "fist", "fists",
    "palm", "palms", "thumb", "thumbs",
    # generic environmental anchors
    "floor", "ground", "ceiling", "wall", "air", "surface", "table",
    "chair", "seat", "stage", "scene", "court", "field", "track",
    # common props / targets
    "ball", "bar", "bat", "racket", "club", "stick", "rope", "net",
    "target", "arrow", "bow", "camera", "microphone", "mirror",
    "drumstick", "drum", "keyboard",
}


def resolve_api_key(cli_api_key: str):
    key = (cli_api_key or "").strip()
    if key:
        return key
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_client(api_key: str, base_url: str):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def choose_stage_sequence(n_temporal: int):
    if n_temporal <= 0:
        return []
    if n_temporal == 1:
        return ["middle"]
    if n_temporal == 2:
        return ["start", "end"]
    if n_temporal == 3:
        return ["start", "middle", "end"]
    base = ["start", "early", "middle", "late", "end"]
    if n_temporal <= len(base):
        return base[:n_temporal]
    return base + [f"phase_{i}" for i in range(1, n_temporal - len(base) + 1)]


def build_user_prompt(label: str, stages, n_spatial: int):
    n_temporal = len(stages)

    if n_temporal == 0:
        # Spatial-only mode
        return f"""Action class: {label}
Spatial descriptor count: {n_spatial}

Return ONLY this JSON object schema:
{{
  "spatial": ["descriptor_1", "descriptor_2"]
}}

Hard output constraints:
- spatial length = {n_spatial}
- each spatial descriptor must be distinctive and class-identifying
- no markdown or code fences
"""

    stage_str = ", ".join(stages)
    return f"""Action class: {label}
Temporal descriptor count: {n_temporal}
Spatial descriptor count: {n_spatial}
Temporal stage order: [{stage_str}]

Return ONLY this JSON object schema:
{{
  "temporal": [
    {{"stage": "{stages[0]}", "text": "..."}}
  ],
  "spatial": ["..."]
}}

Hard output constraints:
- temporal length = {n_temporal}
- spatial length = {n_spatial}
- temporal items must follow exact stage order above
- each temporal item's stage must exactly equal the required token at that index
- each temporal descriptor must capture a visually DISTINCT phase
- no markdown or code fences
"""


def strip_code_fence(text: str):
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return t


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'-]+", text))


def contains_any_token(text: str, tokens: set) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return len(words & tokens) > 0


def contains_anchor_phrase(text: str) -> bool:
    low = text.lower()
    padded = f" {low} "
    if any(f" {w} " in padded for w in ANCHOR_CUES):
        return True
    if any(p in low for p in ["in front of", "next to", "on top of"]):
        return True
    # Accept descriptors that reference a body part or common environmental
    # anchor noun even without an explicit preposition.
    words = set(re.findall(r"[a-zA-Z]+", low))
    return len(words & ANCHOR_NOUNS) > 0


def validate_semantic_rules(temporal_meta, spatial_texts):
    # Uniqueness check
    all_texts = [x["text"].strip().lower() for x in temporal_meta] + \
                [x.strip().lower() for x in spatial_texts]
    if len(all_texts) != len(set(all_texts)):
        raise ValueError("duplicate descriptors detected")

    for i, item in enumerate(temporal_meta):
        text = item["text"].strip()
        low = text.lower()
        if word_count(text) > 15:
            raise ValueError(f"temporal[{i}] exceeds 15 words")
        if contains_any_token(low, VAGUE_WORD_BLACKLIST):
            raise ValueError(f"temporal[{i}] contains vague wording")
        if contains_any_token(low, MENTAL_STATE_BLACKLIST):
            raise ValueError(f"temporal[{i}] contains mental-state wording")
        if not contains_anchor_phrase(low):
            raise ValueError(f"temporal[{i}] missing explicit spatial anchor cue")

    for i, text in enumerate(spatial_texts):
        low = text.strip().lower()
        if word_count(text) > 15:
            raise ValueError(f"spatial[{i}] exceeds 15 words")
        if contains_any_token(low, TEMPORAL_WORD_BLACKLIST):
            raise ValueError(f"spatial[{i}] contains temporal-order words")
        if contains_any_token(low, VAGUE_WORD_BLACKLIST):
            raise ValueError(f"spatial[{i}] contains vague wording")
        if contains_any_token(low, MENTAL_STATE_BLACKLIST):
            raise ValueError(f"spatial[{i}] contains mental-state wording")


def parse_and_validate(raw: str, stages, n_spatial: int):
    obj = json.loads(strip_code_fence(raw))
    if not isinstance(obj, dict):
        raise ValueError("output must be a JSON object")

    temporal = obj.get("temporal", [])
    spatial = obj.get("spatial")
    if not isinstance(temporal, list) or not isinstance(spatial, list):
        raise ValueError("temporal/spatial must both be lists")
    if len(temporal) != len(stages):
        raise ValueError(f"temporal length mismatch: expect {len(stages)} got {len(temporal)}")
    if len(spatial) != n_spatial:
        raise ValueError(f"spatial length mismatch: expect {n_spatial} got {len(spatial)}")

    temporal_texts = []
    temporal_meta = []
    for i, item in enumerate(temporal):
        if not isinstance(item, dict):
            raise ValueError("each temporal item must be an object")
        stage = str(item.get("stage", "")).strip()
        text = str(item.get("text", "")).strip()
        if stage != stages[i]:
            raise ValueError(f"temporal[{i}] stage mismatch: expect {stages[i]} got {stage}")
        if not text:
            raise ValueError(f"temporal[{i}] text empty")
        temporal_texts.append(text)
        temporal_meta.append({"stage": stage, "text": text})

    spatial_texts = [str(x).strip() for x in spatial]
    if any(not s for s in spatial_texts):
        raise ValueError("empty spatial descriptor found")

    validate_semantic_rules(temporal_meta, spatial_texts)

    ordered = temporal_texts + spatial_texts
    return ordered, temporal_meta


def generate_for_label(client, model_name, label, stages, n_spatial,
                       temperature, max_retries, retry_sleep_sec):
    system_prompt = SYSTEM_PROMPT_SPATIAL_ONLY if len(stages) == 0 else SYSTEM_PROMPT_FULL
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(label, stages, n_spatial)},
                ],
                temperature=temperature,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return parse_and_validate(raw, stages, n_spatial)
        except Exception as e:
            last_err = e
            print(f"[WARN] label='{label}' attempt={attempt}/{max_retries} failed: {e}")
            time.sleep(retry_sleep_sec)
    raise RuntimeError(f"failed for label '{label}': {last_err}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ordered temporal/spatial descriptors for Kinetics-400 (SASA-CLIP)."
    )
    p.add_argument("--api-key", type=str, default="",
                   help="OpenAI API key. If empty, read from OPENAI_API_KEY.")
    p.add_argument("--base-url", type=str, default="",
                   help="Optional OpenAI-compatible base URL.")
    p.add_argument("--csv-path", type=str, default="labels/kinetics_400_labels.csv")
    p.add_argument("--id-col", type=str, default="id")
    p.add_argument("--label-col", type=str, default="name")
    p.add_argument("--out-json", type=str, default="descriptors/k400_descriptors.json")
    p.add_argument("--out-meta-json", type=str, default="descriptors/k400_descriptors_meta.json")
    p.add_argument("--model-name", type=str, default="gpt-4.1")
    p.add_argument("--n-temporal", type=int, default=3)
    p.add_argument("--n-spatial", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--retry-sleep-sec", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_temporal < 0 or args.n_spatial < 0:
        raise ValueError("n-temporal and n-spatial must be >= 0")
    if args.n_temporal + args.n_spatial <= 0:
        raise ValueError("total descriptor count must be > 0")

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Use --api-key or set OPENAI_API_KEY.")

    df = pd.read_csv(args.csv_path)
    if args.id_col not in df.columns:
        raise ValueError(f"missing id column: {args.id_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"missing label column: {args.label_col}")

    stages = choose_stage_sequence(args.n_temporal)
    client = build_client(api_key=api_key, base_url=args.base_url)

    result = {}
    meta = {
        "stage_order": stages,
        "n_temporal": args.n_temporal,
        "n_spatial": args.n_spatial,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "classes": {},
    }

    rows = list(df[[args.id_col, args.label_col]].itertuples(index=False, name=None))
    for class_id, label in tqdm(rows, desc="Generating descriptors"):
        ordered, temporal_meta = generate_for_label(
            client=client,
            model_name=args.model_name,
            label=str(label),
            stages=stages,
            n_spatial=args.n_spatial,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_sleep_sec=args.retry_sleep_sec,
        )
        key = str(class_id)
        result[key] = ordered
        meta["classes"][key] = {
            "label": str(label),
            "temporal": temporal_meta,
            "spatial": ordered[len(stages):],
            "ordered_output": ordered,
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_meta_json) or ".", exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(args.out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved ordered descriptors: {args.out_json}")
    print(f"Saved descriptor metadata: {args.out_meta_json}")


if __name__ == "__main__":
    main()
