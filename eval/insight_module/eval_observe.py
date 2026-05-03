"""
观察池模块测试：问题识别准确率
"""
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from insight.core import is_question_by_rule, is_question_with_llm

class ObserveEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase]):
        super().__init__(test_cases)

    def is_question_expected(self, case: TestCase) -> bool:
        """判断预期是否为问题"""
        return 'question-like=true' in case.expected_cards or case.expected_target_pool == 'observe'

    def run(self) -> Dict[str, Any]:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for case in self.test_cases:
            # 只处理观察池相关的用例
            if not ('observe' in case.coverage_path.lower() or case.expected_target_pool == 'observe'):
                continue

            expected_is_question = self.is_question_expected(case)

            # 调用问题识别模块（优先规则，fallback LLM）
            actual_is_question = is_question_by_rule(case.trigger_message)
            if actual_is_question is None:
                actual_is_question = is_question_with_llm(case.trigger_message, case.conversation) or False

            # 统计混淆矩阵
            if expected_is_question and actual_is_question:
                true_positives += 1
            elif not expected_is_question and actual_is_question:
                false_positives += 1
            elif not expected_is_question and not actual_is_question:
                true_negatives += 1
            elif expected_is_question and not actual_is_question:
                false_negatives += 1

            # 保存详细结果
            self.results.append({
                "case_id": case.case_id,
                "scenario": case.scenario,
                "expected_is_question": expected_is_question,
                "actual_is_question": actual_is_question,
                "is_correct": (expected_is_question == actual_is_question)
            })

        # 计算指标
        total_cases = len(self.results)
        accuracy = (true_positives + true_negatives) / total_cases if total_cases > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            "score": self.calculate_score(accuracy),
            "total_cases": total_cases,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "details": self.results
        }

        save_metric_result("observe", result)
        return result

    def calculate_score(self, accuracy: float) -> float:
        """计算模块得分：准确率直接映射到0-100分"""
        return accuracy * 100

if __name__ == "__main__":
    from utils import load_test_cases
    test_cases = load_test_cases()
    tester = ObserveTester(test_cases)
    result = tester.run()
    print(f"观察池模块得分: {result['score']:.2f}/100")
    print(f"问题识别准确率: {result['accuracy']:.4f}")
    print(f"精确率: {result['precision']:.4f}, 召回率: {result['recall']:.4f}, F1: {result['f1_score']:.4f}")