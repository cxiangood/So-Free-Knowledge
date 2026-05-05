"""
信号检测模块评估：精确率、召回率、F1值
直接读取全链路trace结果文件进行评估，无需运行检测模块
"""
import json
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from tqdm import tqdm

class SignalDetectorEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase], trace_file_path: str = "outputs/insight_module_eval/full_pipeline_trace.jsonl"):
        super().__init__(test_cases)
        self.THRESHOLD = 40  # 默认检测阈值
        self.trace_file_path = trace_file_path
        # 预加载所有trace结果
        self.trace_results = self._load_trace_results()

    def _load_trace_results(self) -> Dict[str, Any]:
        """加载全链路trace结果，返回case_id到完整trace数据的映射"""
        trace_map = {}
        try:
            with open(self.trace_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载信号检测trace结果"):
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

    def run(self) -> Dict[str, Any]:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        score_errors = []
        missing_cases = 0

        for case in tqdm(self.test_cases, desc="评估信号检测模块"):
            # 从trace结果中读取实际得分
            trace_data = self.trace_results.get(case.case_id)
            if trace_data is None or 'detect' not in trace_data:
                missing_cases += 1
                continue
            actual_score = trace_data['detect'].get('detect_score', 0.0)

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
        total_valid = len(self.test_cases) - missing_cases
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total_valid if total_valid > 0 else 0
        avg_score_error = sum(score_errors) / len(score_errors) if score_errors else 0

        result = {
            "score": self.calculate_score(precision, recall, f1),
            "total_cases": len(self.test_cases),
            "valid_cases": total_valid,
            "missing_cases": missing_cases,
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
    from eval_utils import load_eval_cases
    test_cases = load_eval_cases()
    tester = SignalDetectorEvaluator(test_cases)
    result = tester.run()
    print(f"信号检测模块得分: {result['score']:.2f}/100")
    print(f"总用例数: {result['total_cases']}, 有效用例: {result['valid_cases']}, 缺失用例: {result['missing_cases']}")
    print(f"精确率: {result['precision']:.4f}, 召回率: {result['recall']:.4f}, F1: {result['f1_score']:.4f}")
