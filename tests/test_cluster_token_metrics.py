import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from wordcloud import WordCloud

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in os.sys.path:
        os.sys.path.insert(0, project_root)

from token_classify.analyzer import SemanticDensityAnalyzer


def load_extract_chat_module():
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "message_archive" / "extract_chat_messages.py",
        root / "message_extract" / "extract_chat_messages.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("extract_chat_messages", path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Cannot find extract_chat_messages.py in message_archive/ or message_extract/.")


def parse_custom_terms(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def select_font_path() -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def aggregate_word_metrics(
    words: List[str],
    semantic_values: List[float],
    entropy_values: List[float],
) -> List[Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "sum_sd": 0.0, "sum_ae": 0.0})
    for w, sd, ae in zip(words, semantic_values, entropy_values):
        normalized_word = " ".join(str(w).split())
        if not normalized_word:
            continue
        rec = stats[normalized_word]
        rec["count"] += 1
        rec["sum_sd"] += float(sd)
        rec["sum_ae"] += float(ae)

    records: List[Dict[str, float]] = []
    for word, rec in stats.items():
        count = int(rec["count"])
        records.append(
            {
                "word": word,
                "count": count,
                "semantic_density": rec["sum_sd"] / max(1, count),
                "attention_entropy": rec["sum_ae"] / max(1, count),
            }
        )
    records.sort(key=lambda x: (-x["count"], x["word"]))
    return records


def run_kmeans(records: List[Dict[str, float]]) -> np.ndarray:
    features = np.array([[r["semantic_density"], r["attention_entropy"]] for r in records], dtype=float)
    model = KMeans(n_clusters=2, init="k-means++", n_init=20, random_state=42)
    return model.fit_predict(features)


def save_word_metrics_csv(records: List[Dict[str, float]], labels: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "count", "semantic_density", "attention_entropy", "cluster"],
        )
        writer.writeheader()
        for rec, label in zip(records, labels):
            writer.writerow(
                {
                    "word": rec["word"],
                    "count": int(rec["count"]),
                    "semantic_density": f"{rec['semantic_density']:.6f}",
                    "attention_entropy": f"{rec['attention_entropy']:.6f}",
                    "cluster": int(label),
                }
            )


def plot_scatter(records: List[Dict[str, float]], labels: np.ndarray, output_path: Path, font_path: Optional[str]) -> None:
    x = np.array([r["semantic_density"] for r in records], dtype=float)
    y = np.array([r["attention_entropy"] for r in records], dtype=float)
    sizes = np.array([max(12.0, 12.0 + 8.0 * math.log1p(r["count"])) for r in records], dtype=float)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=labels, s=sizes, alpha=0.72, cmap="Set2", edgecolors="k", linewidths=0.2)
    plt.xlabel("Semantic Density")
    plt.ylabel("Attention Entropy")
    plt.title("K-means++ Clustering (k=2) on Word Metrics")
    plt.grid(alpha=0.2)
    plt.colorbar(scatter, ticks=[0, 1], label="Cluster")

    top_indices = sorted(range(len(records)), key=lambda i: records[i]["count"], reverse=True)[:20]
    for idx in top_indices:
        word = records[idx]["word"]
        plt.annotate(word, (x[idx], y[idx]), fontsize=8, alpha=0.8)

    if font_path:
        from matplotlib import font_manager

        font_prop = font_manager.FontProperties(fname=font_path)
        for text in plt.gca().texts:
            text.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_cluster_freq(records: List[Dict[str, float]], labels: np.ndarray) -> Dict[int, Dict[str, int]]:
    cluster_freq: Dict[int, Dict[str, int]] = {0: {}, 1: {}}
    for rec, label in zip(records, labels):
        cluster_freq[int(label)][rec["word"]] = int(rec["count"])
    return cluster_freq


def save_wordcloud(freq: Dict[str, int], output_path: Path, title: str, font_path: Optional[str]) -> None:
    if not freq:
        return
    wc = WordCloud(
        width=1400,
        height=900,
        background_color="white",
        collocations=False,
        font_path=font_path,
    ).generate_from_frequencies(freq)
    wc.to_file(str(output_path))

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    panel_path = output_path.with_name(output_path.stem + "_panel.png")
    plt.savefig(panel_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster word metrics (semantic density + attention entropy) with K-means++.")
    parser.add_argument(
        "--input",
        default="message_archive/20260425T130609Z/messages.jsonl",
        help="Path to messages json/jsonl for testing.",
    )
    parser.add_argument("--include-types", default="text,post")
    parser.add_argument("--custom-terms", default="")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--ngram-min-count", type=int, default=2)
    parser.add_argument("--ngram-min-pmi", type=float, default=4.0)
    parser.add_argument("--ngram-max-n", type=int, default=3)
    parser.add_argument("--max-messages", type=int, default=-1, help="Use -1 for all messages.")
    parser.add_argument("--min-word-freq", type=int, default=2)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.output_dir) if args.output_dir else Path("token_classify") / "outputs" / (
        "cluster_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_module = load_extract_chat_module()
    records = extract_module.load_records(input_path)
    include_types = {x.strip().lower() for x in args.include_types.split(",") if x.strip()} or {"text", "post"}
    plain_messages = extract_module.extract_plain_messages(records, include_types=include_types)
    if args.max_messages > 0:
        plain_messages = plain_messages[: args.max_messages]
    merged_text = "\n".join(plain_messages)

    analyzer = SemanticDensityAnalyzer(
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap_ratio,
        custom_terms=parse_custom_terms(args.custom_terms),
        ngram_min_count=args.ngram_min_count,
        ngram_min_pmi=args.ngram_min_pmi,
        ngram_max_n=args.ngram_max_n,
    )

    words, semantic_values = analyzer.semantic_density(merged_text)
    words2, entropy_values = analyzer.attention_entropy(merged_text)
    if words != words2:
        raise RuntimeError("Word list mismatch between semantic_density and attention_entropy outputs.")

    agg_records = aggregate_word_metrics(words, semantic_values, entropy_values)
    agg_records = [r for r in agg_records if int(r["count"]) >= args.min_word_freq]
    if len(agg_records) < 2:
        raise RuntimeError("Not enough words to cluster after filtering. Lower --min-word-freq or provide more data.")

    labels = run_kmeans(agg_records)
    font_path = select_font_path()

    csv_path = out_dir / "word_metrics_clustered.csv"
    scatter_path = out_dir / "cluster_scatter.png"
    wc0_path = out_dir / "cluster_0_wordcloud.png"
    wc1_path = out_dir / "cluster_1_wordcloud.png"
    summary_path = out_dir / "summary.json"

    save_word_metrics_csv(agg_records, labels, csv_path)
    plot_scatter(agg_records, labels, scatter_path, font_path)

    cluster_freq = build_cluster_freq(agg_records, labels)
    save_wordcloud(cluster_freq.get(0, {}), wc0_path, "Cluster 0 WordCloud", font_path)
    save_wordcloud(cluster_freq.get(1, {}), wc1_path, "Cluster 1 WordCloud", font_path)

    summary = {
        "input_file": str(input_path),
        "messages_used": len(plain_messages),
        "word_instances": len(words),
        "unique_words_after_filter": len(agg_records),
        "cluster_counts": {
            "0": int(np.sum(labels == 0)),
            "1": int(np.sum(labels == 1)),
        },
        "outputs": {
            "metrics_csv": str(csv_path),
            "scatter": str(scatter_path),
            "wordcloud_cluster_0": str(wc0_path),
            "wordcloud_cluster_1": str(wc1_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
