#!/usr/bin/env python3
"""测试KeywordClassifier关键词分类功能"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_classify.keyword_classifier import KeywordClassifier


def test_keyword_classifier():
    """测试关键词分类器"""
    print("正在初始化关键词分类器...")

    try:
        # 创建分类器实例
        classifier = KeywordClassifier()
        print("分类器初始化成功")
    except Exception as e:
        print(f"分类器初始化失败: {e}")
        return False

    # 测试用例
    test_cases = [
        {
            "keywords": ["人工智能", "机器学习", "深度学习", "区块链", "比特币"],
            "contexts": [
                "人工智能是计算机科学的一个分支，包括机器学习和深度学习等领域。",
                "区块链是一种分布式账本技术，比特币是其最知名的应用。",
                "深度学习在图像识别和自然语言处理方面有广泛应用。"
            ]
        },
        {
            "keywords": ["COVID-19", "疫苗", "核酸检测", "健康码"],
            "contexts": [
                "COVID-19疫情期间，疫苗接种和核酸检测成为防控的重要手段。",
                "健康码用于记录个人健康状态和出行轨迹。"
            ]
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== 测试用例 {i} ===")
        keywords = test_case["keywords"]
        contexts = test_case["contexts"]

        print(f"关键词: {', '.join(keywords)}")
        print(f"上下文数量: {len(contexts)}")
        print("正在分类...")

        try:
            result = classifier.classify_group_keywords(keywords, contexts)
            print("分类成功！结果：")
            # print(result)
            for keyword, info in result.items():
                print(f"  - {keyword}: type={info['type']}, sense={info['sense']}")
        except Exception as e:
            print(f"分类失败: {e}")
            continue

    return True


if __name__ == "__main__":
    success = test_keyword_classifier()
    sys.exit(0 if success else 1)