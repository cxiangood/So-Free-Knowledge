"""
路由模块测试：路由准确率
"""
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from insight.core.route import route_cards
from insight.core.lift import lift_candidates
from insight.core.detect import DetectionResult

class RouterEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase]):
        super().__init__(test_cases)

    def parse_expected_route(self, expected_target_pool: str) -> str:
        """解析预期路由目标"""
        if not expected_target_pool or expected_target_pool == '无 decisions':
            return 'finalize'
        return expected_target_pool.lower()

    def run(self) -> Dict[str, Any]:
        correct_routes = 0
        route_confusion = {}  # 统计错误路由的分布

        for case in self.test_cases:
            expected_route = self.parse_expected_route(case.expected_target_pool)

            # 构造消息列表
            messages = case.conversation + [case.trigger_message]
            # 调用语义升维模块
            lift_result = lift_candidates(messages)
            if not lift_result.cards:
                actual_route = 'finalize'
            else:
                # 调用路由模块
                route_result = route_cards(lift_result.cards)
                actual_route = route_result[0].target if route_result else 'finalize'
            actual_route = actual_route.lower()

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
        accuracy = correct_routes / len(self.test_cases) if self.test_cases else 0

        result = {
            "score": self.calculate_score(accuracy),
            "total_cases": len(self.test_cases),
            "correct_routes": correct_routes,
            "incorrect_routes": len(self.test_cases) - correct_routes,
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
    from utils import load_test_cases
    test_cases = load_test_cases()
    tester = RouterTester(test_cases)
    result = tester.run()
    print(f"路由模块得分: {result['score']:.2f}/100")
    print(f"路由准确率: {result['accuracy']:.4f}")
    if result['route_confusion']:
        print("错误路由分布:", result['route_confusion'])