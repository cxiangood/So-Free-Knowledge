"""
观察池模块评估：问题识别准确率
直接读取全链路trace结果文件进行评估，无需运行模块
"""
import json
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from tqdm import tqdm

class ObserveEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase], trace_file_path: str = "outputs/insight_module_eval/full_pipeline_trace.jsonl"):
        super().__init__(test_cases)
        self.trace_file_path = trace_file_path
        # 预加载所有trace结果
        self.trace_results = self._load_trace_results()

    def _load_trace_results(self) -> Dict[str, Any]:
        """加载全链路trace结果，返回case_id到完整trace数据的映射"""
        trace_map = {}
        try:
            with open(self.trace_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载观察池trace结果"):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    case_id = data.get("case_id")
                    if case_id:
                        trace_map[case_id] = data
        except FileNotFoundError:
            print(f"警告：trace文件 {self.trace_file_path} 不存在，将使用空结果")
        return trace_map

    def is_question_expected(self, case: TestCase) -> bool:
        """判断预期是否为问题"""
        return 'question-like=true' in case.expected_cards or case.expected_target_pool == 'observe'

    def run(self) -> Dict[str, Any]:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        missing_cases = 0

        for case in tqdm(self.test_cases, desc="评估观察池模块"):
            # 只处理观察池相关的用例
            if not ('observe' in case.coverage_path.lower() or case.expected_target_pool == 'observe'):
                continue

            expected_is_question = self.is_question_expected(case)

            # 从trace结果中读取实际结果
            trace_data = self.trace_results.get(case.case_id)
            if trace_data is None:
                missing_cases += 1
                continue

            # 从lift结果的decision_signals中获取是否为问题的判断（更准确）
            actual_is_question = False
            if 'lift' in trace_data and trace_data['lift'] is not None and trace_data['lift'].get('cards'):
                card = trace_data['lift']['cards'][0]
                decision_signals = card.get('decision_signals', {})
                actual_is_question = decision_signals.get('has_question', 0.0) >= 0.5

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
            "missing_cases": missing_cases,
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
    from eval_utils import load_eval_cases
    test_cases = load_eval_cases()
    evaluator = ObserveEvaluator(test_cases)
    result = evaluator.run()
    print(f"观察池模块得分: {result['score']:.2f}/100")
    print(f"总用例数: {result['total_cases']}, 缺失用例: {result['missing_cases']}")
    print(f"问题识别准确率: {result['accuracy']:.4f}")
    print(f"精确率: {result['precision']:.4f}, 召回率: {result['recall']:.4f}, F1: {result['f1_score']:.4f}")
