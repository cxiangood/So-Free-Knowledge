"""
信号检测模块测试：精确率、召回率、F1值
"""
from typing import List, Dict, Any
from eval.insight_module.eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from insight.core.detect import detect_candidates
from tqdm import tqdm

class SignalDetectorEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase]):
        super().__init__(test_cases)
        self.THRESHOLD = 40  # 默认检测阈值

    def run(self) -> Dict[str, Any]:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        score_errors = []

        for case in tqdm(self.test_cases):
            # 构造完整消息列表：上下文 + 触发消息
            all_messages = case.conversation
            # 调用信号检测模块
            result = detect_candidates(all_messages)
            # print(result)
            actual_score = result.value_score
            score_errors.append(abs(actual_score - (case.expected_detect_score_min + case.expected_detect_score_max) / 2))

            # 判断是否为高价值信号（预期得分>阈值视为正样本）
            expected_is_high_value = case.expected_detect_score_min >= self.THRESHOLD
            actual_is_high_value = actual_score >= self.THRESHOLD

            # 统计混淆矩阵
            if expected_is_high_value and actual_is_high_value:
                true_positives += 1
            elif not expected_is_high_value and actual_is_high_value:
                false_positives += 1
            elif not expected_is_high_value and not actual_is_high_value:
                true_negatives += 1
            elif expected_is_high_value and not actual_is_high_value:
                false_negatives += 1

            # 保存详细结果
            self.results.append({
                "case_id": case.case_id,
                "scenario": case.scenario,
                "expected_score_range": f"{case.expected_detect_score_min}-{case.expected_detect_score_max}",
                "actual_score": actual_score,
                "expected_high_value": expected_is_high_value,
                "actual_high_value": actual_is_high_value,
                "is_correct": (expected_is_high_value == actual_is_high_value)
            })

        # 计算指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(self.test_cases) if self.test_cases else 0
        avg_score_error = sum(score_errors) / len(score_errors) if score_errors else 0

        result = {
            "score": self.calculate_score(precision, recall, f1),
            "total_cases": len(self.test_cases),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "avg_score_error": round(avg_score_error, 2),
            "details": self.results
        }

        save_metric_result("signal_detector", result)
        return result

    def calculate_score(self, precision: float, recall: float, f1: float) -> float:
        """
        计算模块得分：
        - 精确率占40%
        - 召回率占40%
        - F1值占20%
        """
        return (precision * 0.4 + recall * 0.4 + f1 * 0.2) * 100

if __name__ == "__main__":
    from eval.insight_module.eval_utils import load_eval_cases
    test_cases = load_eval_cases(csv_path="datas/chat_test_optimized_desensitized_corrected.csv")
    # print(test_cases)
    tester = SignalDetectorEvaluator(test_cases)
    result = tester.run()
    print(f"信号检测模块得分: {result['score']:.2f}/100")
    print(f"精确率: {result['precision']:.4f}, 召回率: {result['recall']:.4f}, F1: {result['f1_score']:.4f}")