from token_classify.domain_tokenizer import DomainAdaptiveTokenizer, tokenize_text

__all__ = ["SemanticDensityAnalyzer", "DomainAdaptiveTokenizer", "tokenize_text", "classify_text", "classify"]


def __getattr__(name: str):
    if name == "SemanticDensityAnalyzer":
        from token_classify.analyzer import SemanticDensityAnalyzer

        return SemanticDensityAnalyzer
    if name == "classify_text":
        from token_classify.classify import classify_text

        return classify_text
    if name == "classify":
        from token_classify.classify import classify

        return classify
    raise AttributeError(name)
