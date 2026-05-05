"""
路由模块评估：路由准确率
直接读取全链路trace结果文件进行评估，无需运行模块
"""
import json
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from tqdm import tqdm

class RouterEvaluator(BaseModuleEvaluator):
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
                for line in tqdm(f, desc="加载路由trace结果"):
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

    def parse_expected_route(self, expected_target_pool: str) -> str:
        """解析预期路由目标"""
        if not expected_target_pool or expected_target_pool == '无 decisions':
            return 'finalize'
        return expected_target_pool.lower()

    def run(self) -> Dict[str, Any]:
        correct_routes = 0
        route_confusion = {}  # 统计错误路由的分布
        missing_cases = 0

        for case in tqdm(self.test_cases, desc="评估路由模块"):
            expected_route = self.parse_expected_route(case.expected_target_pool)

            # 从trace结果中读取实际路由
            trace_data = self.trace_results.get(case.case_id)
            if trace_data is None or 'route' not in trace_data:
                missing_cases += 1
                actual_route = 'missing'
            else:
                route_result = trace_data['route']
                if route_result is None:
                    decisions = []
                else:
                    decisions = route_result.get('decisions', [])
                if not decisions:
                    actual_route = 'finalize'
                else:
                    # 取第一个路由决策
                    actual_route = decisions[0].get('target_pool', 'finalize').lower()

            # 判断是否正确
            is_correct = expected_route == actual_route
            if is_correct:
                correct_routes += 1
            else:
                # 统计错误分布
                key = f"{expected_route} → {actual_route}"
                route_confusion[key] = route_confusion.get(key, 0) + 1

            # 保存详细结果
            self.results.append({
                "case_id": case.case_id,
                "scenario": case.scenario,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "is_correct": is_correct
            })

        # 计算指标
        valid_cases = len(self.test_cases) - missing_cases
        accuracy = correct_routes / valid_cases if valid_cases > 0 else 0

        result = {
            "score": self.calculate_score(accuracy),
            "total_cases": len(self.test_cases),
            "valid_cases": valid_cases,
            "missing_cases": missing_cases,
            "correct_routes": correct_routes,
            "incorrect_routes": valid_cases - correct_routes,
            "accuracy": round(accuracy, 4),
            "route_confusion": route_confusion,
            "details": self.results
        }

        save_metric_result("router", result)
        return result

    def calculate_score(self, accuracy: float) -> float:
        """计算模块得分：准确率直接映射到0-100分"""
        return accuracy * 100

if __name__ == "__main__":
    from eval_utils import load_eval_cases
    test_cases = load_eval_cases()
    evaluator = RouterEvaluator(test_cases)
    result = evaluator.run()
    print(f"路由模块得分: {result['score']:.2f}/100")
    print(f"总用例数: {result['total_cases']}, 有效用例: {result['valid_cases']}, 缺失用例: {result['missing_cases']}")
    print(f"路由准确率: {result['accuracy']:.4f}")
    if result['route_confusion']:
        print("错误路由分布:", result['route_confusion'])
