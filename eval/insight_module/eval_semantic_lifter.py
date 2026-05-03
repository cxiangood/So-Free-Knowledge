"""
语义升维模块测试：字段提取准确率
"""
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from insight.core.lift import lift_candidates
from insight.core.detect import DetectionResult

class SemanticLifterEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase]):
        super().__init__(test_cases)

    def parse_expected_fields(self, expected_cards: str, target_pool: str) -> Dict[str, Any]:
        """解析预期字段"""
        expected_fields = {}
        if target_pool == 'knowledge':
            # 知识卡片预期字段：title, summary, tags
            if 'title=' in expected_cards:
                expected_fields['title'] = expected_cards.split('title=')[1].split(';')[0].strip()
            if 'summary' in expected_cards:
                expected_fields['summary'] = expected_cards.split('summary')[1].split(';')[0].strip('= ')
            if 'tags' in expected_cards:
                expected_fields['tags'] = expected_cards.split('tags')[1].split(';')[0].strip('= ').split('/')
        elif target_pool == 'task':
            # 任务卡片预期字段：title, names, time
            if 'title=' in expected_cards:
                expected_fields['title'] = expected_cards.split('title=')[1].split(';')[0].strip()
            if 'names=' in expected_cards:
                expected_fields['assignees'] = expected_cards.split('names=')[1].split(']')[0].strip('[]').split(',')
            if 'time=' in expected_cards:
                expected_fields['deadline'] = expected_cards.split('time=')[1].strip()
        elif target_pool == 'observe':
            # 观察卡片预期字段：problem, question-like
            if 'problem=' in expected_cards:
                expected_fields['problem'] = expected_cards.split('problem=')[1].split(';')[0].strip()
            if 'question-like=' in expected_cards:
                parts = expected_cards.split('question-like=')
                if len(parts) > 1:
                    expected_fields['is_question'] = parts[1].split(';')[0].strip() == 'true'
        return expected_fields

    def calculate_field_accuracy(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """计算单条用例的字段准确率"""
        if not expected:
            return 1.0  # 无预期字段时默认正确
        correct_fields = 0
        for field, expected_value in expected.items():
            actual_value = actual.get(field)
            if isinstance(expected_value, list):
                # 列表类型字段匹配（忽略顺序）
                if actual_value and set(actual_value) == set(expected_value):
                    correct_fields += 1
            elif isinstance(expected_value, str):
                # 字符串类型字段包含匹配
                if actual_value and expected_value in str(actual_value):
                    correct_fields += 1
            elif actual_value == expected_value:
                # 其他类型精确匹配
                correct_fields += 1
        return correct_fields / len(expected)

    def run(self) -> Dict[str, Any]:
        total_accuracy = 0.0
        field_level_correct = {}
        field_level_total = {}

        for case in self.test_cases:
            # 跳过无预期字段的用例
            if not case.expected_cards or case.expected_cards == '[]':
                continue

            # 构造消息列表
            messages = case.conversation + [case.trigger_message]
            # 调用语义升维模块
            result = lift_candidates(messages)
            actual_fields = {}
            if result.cards:
                card = result.cards[0]
                actual_fields = {
                    'title': card.title,
                    'summary': card.summary,
                    'tags': card.tags,
                    'assignees': card.participants,
                    'deadline': card.times,
                    'problem': card.problem,
                    'is_question': card.decision_signals.get('has_question', 0) >= 0.5 if card.decision_signals else False
                }
            expected_fields = self.parse_expected_fields(case.expected_cards, case.expected_target_pool)

            # 计算字段准确率
            case_accuracy = self.calculate_field_accuracy(actual_fields, expected_fields)
            total_accuracy += case_accuracy

            # 统计字段级别准确率
            for field in expected_fields:
                field_level_total[field] = field_level_total.get(field, 0) + 1
                actual_value = actual_fields.get(field)
                expected_value = expected_fields[field]
                if isinstance(expected_value, list) and actual_value and set(actual_value) == set(expected_value):
                    field_level_correct[field] = field_level_correct.get(field, 0) + 1
                elif isinstance(expected_value, str) and actual_value and expected_value in str(actual_value):
                    field_level_correct[field] = field_level_correct.get(field, 0) + 1
                elif actual_value == expected_value:
                    field_level_correct[field] = field_level_correct.get(field, 0) + 1

            # 保存详细结果
            self.results.append({
                "case_id": case.case_id,
                "scenario": case.scenario,
                "target_pool": case.expected_target_pool,
                "expected_fields": expected_fields,
                "actual_fields": actual_fields,
                "accuracy": round(case_accuracy, 4)
            })

        # 计算整体指标
        avg_accuracy = total_accuracy / len(self.results) if self.results else 0
        field_accuracies = {
            field: round(field_level_correct.get(field, 0) / field_level_total.get(field, 1), 4)
            for field in field_level_total
        }

        result = {
            "score": self.calculate_score(avg_accuracy),
            "total_cases": len(self.results),
            "average_accuracy": round(avg_accuracy, 4),
            "field_level_accuracies": field_accuracies,
            "details": self.results
        }

        save_metric_result("semantic_lifter", result)
        return result

    def calculate_score(self, avg_accuracy: float) -> float:
        """计算模块得分：准确率直接映射到0-100分"""
        return avg_accuracy * 100

if __name__ == "__main__":
    from utils import load_test_cases
    test_cases = load_test_cases()
    tester = SemanticLifterTester(test_cases)
    result = tester.run()
    print(f"语义升维模块得分: {result['score']:.2f}/100")
    print(f"平均字段提取准确率: {result['average_accuracy']:.4f}")
    print("字段级别准确率:", result['field_level_accuracies'])