# history_router_eval.py
#
# pip install -U openai datasets tqdm
#
# export OPENAI_API_KEY=...
# python history_router_eval.py
#
# 100 examples:
#   - 50 history-dependent: CANARD/CQR
#   - 50 history-independent: SQuAD + irrelevant conversational history
#
# Methods:
#   1. always_all
#   2. binary_gate        -> exactly 0 / 1
#   3. history_select     -> 0 or e.g. 1,3
#
# GPT-6 Luna, reasoning=none

import os
import re
import json
import time
import random
import statistics
from collections import Counter

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


MODEL = "gpt-6-luna"
SEED = 42

N_POS = 50
N_NEG = 50

MAX_HISTORY_TURNS = 8

random.seed(SEED)

client = OpenAI()


# ============================================================
# Utils
# ============================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on",
    "at", "for", "with", "by", "from", "and", "or",
    "but", "if", "then", "than", "as",
    "what", "who", "where", "when", "why", "how",
    "which", "do", "does", "did", "has", "have", "had",
    "this", "that", "these", "those",
    "he", "she", "it", "they", "them", "his", "her",
    "their", "its", "you", "your", "i", "we"
}


def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def words(s):
    return set(
        x for x in re.findall(r"[a-z0-9]+", s.lower())
        if len(x) > 2 and x not in STOPWORDS
    )


def call_luna(prompt, max_output_tokens=16):
    max_output_tokens = max(16, max_output_tokens)

    t0 = time.perf_counter()

    r = client.responses.create(
        model=MODEL,
        reasoning={"effort": "none"},
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    text = r.output_text.strip()
    latency = time.perf_counter() - t0

    usage = getattr(r, "usage", None)

    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None

    return {
        "text": text,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

# ============================================================
# Build positive data
# ============================================================

def get_cqr():
    ds = load_dataset(
        "redis/langcache-cqr-v1",
        split="test"
    )

    rows = []

    for x in ds:

        # Prefer actual CANARD samples
        source = str(x.get("dataset", "")).lower()

        if source and "canard" not in source:
            continue

        history = x["context"]
        question = x["question"]
        rewrite = x["rewrite"]

        if not history:
            continue

        history = history[-MAX_HISTORY_TURNS:]

        # Must genuinely change when rewritten
        if normalize(question) == normalize(rewrite):
            continue

        q_words = words(question)
        rewrite_words = words(rewrite)

        # Terms introduced by contextual resolution
        novel = rewrite_words - q_words

        if not novel:
            continue

        gold_turns = []

        for i, turn in enumerate(history):
            if words(turn) & novel:
                gold_turns.append(i + 1)

        # Need at least one identifiable relevant history turn
        if not gold_turns:
            continue

        rows.append({
            "type": "multi",
            "history": history,
            "question": question,
            "rewrite": rewrite,
            "gold_need_history": 1,
            "gold_turns": gold_turns,
        })

    random.shuffle(rows)

    return rows[:N_POS]


# ============================================================
# Build negative data
# ============================================================

def get_negatives(history_pool):
    squad = load_dataset(
        "rajpurkar/squad",
        split="validation"
    )

    indices = list(range(len(squad)))
    random.shuffle(indices)

    rows = []

    for idx in indices:
        x = squad[idx]

        q = x["question"].strip()

        if len(q) < 10:
            continue

        # Add irrelevant conversational history deliberately.
        donor = random.choice(history_pool)

        fake_history = donor["history"]

        rows.append({
            "type": "single",
            "history": fake_history,
            "question": q,
            "rewrite": q,
            "gold_need_history": 0,
            "gold_turns": [],
        })

        if len(rows) >= N_NEG:
            break

    return rows


# ============================================================
# Prompt formatting
# ============================================================

def format_history(history):
    return "\n".join(
        f"{i+1}. {turn}"
        for i, turn in enumerate(history)
    )


# ============================================================
# Method 1: Always all
# ============================================================

def predict_always_all(x):

    n = len(x["history"])

    return {
        "need_history": 1 if n else 0,
        "selected": list(range(1, n + 1)),
        "raw": "ALL",
        "latency": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


# ============================================================
# Method 2: Binary 0/1 gate
# ============================================================

BINARY_PROMPT = """Determine whether the CURRENT QUESTION requires information from the conversation HISTORY to understand what the user is asking.

Output exactly one character:

1 = history is required
0 = current question can be understood independently of history

Do not explain.
Do not output punctuation.
Do not output anything except 0 or 1.

HISTORY:
{history}

CURRENT QUESTION:
{question}

OUTPUT:"""


def predict_binary(x):

    prompt = BINARY_PROMPT.format(
        history=format_history(x["history"]),
        question=x["question"]
    )

    r = call_luna(prompt, max_output_tokens=16)

    raw = r["text"].strip()

    m = re.search(r"[01]", raw)

    if not m:
        pred = 0
    else:
        pred = int(m.group(0))

    # Binary gate means:
    # if history needed -> use ALL history
    # otherwise -> use nothing
    selected = (
        list(range(1, len(x["history"]) + 1))
        if pred == 1
        else []
    )

    return {
        "need_history": pred,
        "selected": selected,
        "raw": raw,
        "latency": r["latency"],
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
    }


# ============================================================
# Method 3: History turn selection
# ============================================================

SELECT_PROMPT = """Select which HISTORY turns are necessary to understand the CURRENT QUESTION.

Rules:
- If no history is needed, output exactly:
0

- Otherwise output only the turn numbers separated by commas.
Example:
1,3

- Select only turns that are actually necessary.
- Do not explain.
- Do not output brackets.
- Do not output any other text.

HISTORY:
{history}

CURRENT QUESTION:
{question}

OUTPUT:"""


def parse_selection(raw, n_history):

    raw = raw.strip()

    if raw == "0":
        return []

    nums = re.findall(r"\d+", raw)

    selected = []

    for n in nums:
        n = int(n)

        if 1 <= n <= n_history:
            selected.append(n)

    return sorted(set(selected))


def predict_select(x):

    prompt = SELECT_PROMPT.format(
        history=format_history(x["history"]),
        question=x["question"]
    )

    r = call_luna(prompt, max_output_tokens=16)  # 12 -> 16

    selected = parse_selection(
        r["text"],
        len(x["history"])
    )

    return {
        "need_history": int(len(selected) > 0),
        "selected": selected,
        "raw": r["text"],
        "latency": r["latency"],
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
    }

# ============================================================
# Metrics
# ============================================================

def safe_div(a, b):
    return a / b if b else 0.0


def binary_metrics(results):

    tp = tn = fp = fn = 0

    for r in results:

        y = r["gold_need_history"]
        p = r["pred"]["need_history"]

        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1

    acc = safe_div(tp + tn, len(results))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    f1 = safe_div(
        2 * precision * recall,
        precision + recall
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def selection_metrics(results):

    tp = fp = fn = 0

    exact = 0
    positive_count = 0

    selected_counts = []

    for r in results:

        pred = set(r["pred"]["selected"])
        gold = set(r["gold_turns"])

        selected_counts.append(len(pred))

        # Turn-level metrics only meaningful for positive samples
        if r["gold_need_history"] == 1:

            positive_count += 1

            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)

            if pred == gold:
                exact += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    f1 = safe_div(
        2 * precision * recall,
        precision + recall
    )

    return {
        "turn_precision": precision,
        "turn_recall": recall,
        "turn_f1": f1,
        "turn_exact": safe_div(exact, positive_count),
        "avg_selected_turns": statistics.mean(selected_counts),
    }


def cost_metrics(results):

    latencies = [
        r["pred"]["latency"]
        for r in results
    ]

    input_tokens = [
        r["pred"]["input_tokens"]
        for r in results
        if r["pred"]["input_tokens"] is not None
    ]

    output_tokens = [
        r["pred"]["output_tokens"]
        for r in results
        if r["pred"]["output_tokens"] is not None
    ]

    return {
        "mean_latency": statistics.mean(latencies),
        "mean_input_tokens":
            statistics.mean(input_tokens)
            if input_tokens else 0,
        "mean_output_tokens":
            statistics.mean(output_tokens)
            if output_tokens else 0,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_method(name, data, fn):

    results = []

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    for i, x in enumerate(tqdm(data)):

        pred = fn(x)

        results.append({
            **x,
            "pred": pred,
        })

    bm = binary_metrics(results)
    sm = selection_metrics(results)
    cm = cost_metrics(results)

    metrics = {
        **bm,
        **sm,
        **cm,
    }

    return results, metrics


# ============================================================
# Main
# ============================================================

def main():

    print("Loading CQR/CANARD...")
    positives = get_cqr()

    if len(positives) < N_POS:
        raise RuntimeError(
            f"Only found {len(positives)} positive samples"
        )

    print("Building standalone negatives...")
    negatives = get_negatives(positives)

    data = positives + negatives

    random.shuffle(data)

    print()
    print(f"Total : {len(data)}")
    print(f"multi : {sum(x['gold_need_history'] for x in data)}")
    print(f"single: {sum(1-x['gold_need_history'] for x in data)}")

    methods = {
        "always_all": predict_always_all,
        "binary_gate": predict_binary,
        "history_select": predict_select,
    }

    all_metrics = {}
    all_results = {}

    for name, fn in methods.items():

        results, metrics = evaluate_method(
            name,
            data,
            fn
        )

        all_results[name] = results
        all_metrics[name] = metrics

    print("\n\n")
    print("=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    header = (
        f"{'method':<18}"
        f"{'bin_acc':>10}"
        f"{'bin_f1':>10}"
        f"{'turn_P':>10}"
        f"{'turn_R':>10}"
        f"{'turn_F1':>10}"
        f"{'exact':>10}"
        f"{'avg_turn':>10}"
        f"{'latency':>10}"
    )

    print(header)

    for name, m in all_metrics.items():

        print(
            f"{name:<18}"
            f"{m['accuracy']:>10.3f}"
            f"{m['f1']:>10.3f}"
            f"{m['turn_precision']:>10.3f}"
            f"{m['turn_recall']:>10.3f}"
            f"{m['turn_f1']:>10.3f}"
            f"{m['turn_exact']:>10.3f}"
            f"{m['avg_selected_turns']:>10.2f}"
            f"{m['mean_latency']:>10.3f}"
        )

    with open(
        "history_router_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "metrics": all_metrics,
                "results": all_results
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("saved: history_router_results.json")


if __name__ == "__main__":
    main()
