from token_classify.domain_tokenizer import DomainAdaptiveTokenizer, tokenize_text

__all__ = ["SemanticDensityAnalyzer", "DomainAdaptiveTokenizer", "tokenize_text"]


def __getattr__(name: str):
    if name == "SemanticDensityAnalyzer":
        from token_classify.analyzer import SemanticDensityAnalyzer

        return SemanticDensityAnalyzer
    raise AttributeError(name)
