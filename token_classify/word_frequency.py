#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词频统计模块 - 统计词频并按词频排序
"""
from collections import Counter
from typing import List, Tuple, Optional

def calculate_word_frequency(tokens: List[str], top_n: Optional[int] = None, stop_words: Optional[List[str]] = None) -> List[Tuple[str, int]]:
    """
    统计词频并按词频降序排序

    Args:
        tokens: 分词结果列表（已过滤符号）
        top_n: 返回前N个高频词，None表示返回所有
        stop_words: 自定义停止词列表，这些词会被排除统计

    Returns:
        按词频降序排列的 (词, 频次) 列表
    """
    # 应用停止词过滤
    if stop_words:
        stop_words_set = set(stop_words)
        tokens = [token for token in tokens if token not in stop_words_set]

    # 统计词频
    counter = Counter(tokens)

    # 按词频降序排序
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # 返回前N个
    if top_n is not None and top_n > 0:
        sorted_items = sorted_items[:top_n]

    return sorted_items

def get_top_keywords(tokens: List[str], top_k: int = 20, stop_words: Optional[List[str]] = None) -> List[str]:
    """
    获取前K个高频关键词

    Args:
        tokens: 分词结果列表
        top_k: 返回关键词数量
        stop_words: 停止词列表

    Returns:
        关键词列表
    """
    word_freq = calculate_word_frequency(tokens, top_n=top_k, stop_words=stop_words)
    return [word for word, freq in word_freq]

if __name__ == "__main__":
    # 测试用例
    test_tokens = [
        "人工智能", "机器学习", "深度学习", "神经网络", "人工智能",
        "机器学习", "人工智能", "大数据", "云计算", "大数据",
        "人工智能", "机器学习", "深度学习", "人工智能", "机器学习",
        "Python", "Python", "Python", "编程", "编程", "算法", "算法", "算法", "算法"
    ]

    # 测试基本词频统计
    freq = calculate_word_frequency(test_tokens)
    print("词频统计结果:")
    for word, count in freq:
        print(f"{word}: {count}")

    # 测试前N个
    print("\n前5个高频词:")
    top5 = calculate_word_frequency(test_tokens, top_n=5)
    for word, count in top5:
        print(f"{word}: {count}")

    # 测试停止词
    stop_words = ["Python", "编程", "算法"]
    print("\n过滤停止词后的词频:")
    freq_without_stop = calculate_word_frequency(test_tokens, stop_words=stop_words)
    for word, count in freq_without_stop:
        print(f"{word}: {count}")

    # 测试获取关键词列表
    print("\n前3个关键词:")
    keywords = get_top_keywords(test_tokens, top_k=3)
    print(keywords)
