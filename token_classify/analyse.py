import warnings

from token_classify.analyzer import SemanticDensityAnalyzer

warnings.filterwarnings("ignore")


def demo_analysis():
    analyzer = SemanticDensityAnalyzer(
        model_name="bert-base-chinese",
        chunk_size=256,
        overlap_ratio=0.2,
        custom_terms=["人工智能", "风控策略", "A/B测试", "召回率"],
        ngram_min_count=2,
        ngram_min_pmi=2.5,
        ngram_max_n=3,
    )

    test_text = (
        "我们在AI风控系统里上线了新的A/B测试方案，"
        "人工智能模型对黑产账号识别效果提升明显，"
        "recall从0.71提升到0.79，线上QPS保持稳定。"
    )

    density_words, density_values = analyzer.semantic_density(test_text)
    entropy_words, entropy_values = analyzer.attention_entropy(test_text)

    print("=" * 80)
    print("Token-Classify Demo")
    print("=" * 80)
    print(f"text: {test_text}")
    print("-" * 80)
    print(f"semantic_density items: {len(density_words)}")
    print(f"attention_entropy items: {len(entropy_words)}")
    print("-" * 80)
    print("semantic_density (top 10):")
    for word, value in list(zip(density_words, density_values))[:10]:
        print(f"{word}: {value:.4f}")
    print("-" * 80)
    print("attention_entropy (top 10):")
    for word, value in list(zip(entropy_words, entropy_values))[:10]:
        print(f"{word}: {value:.4f}")


if __name__ == "__main__":
    demo_analysis()
