"""
语义升维模块评估：字段提取准确率
直接读取全链路trace结果文件进行评估，无需运行模块
使用BGE语义相似度匹配代替简单字符串匹配
"""
import json
import os
import numpy as np
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

class SemanticLifterEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase], trace_file_path: str = "outputs/insight_module_eval/insight_full_pipeline_chat_test_2.jsonl", similarity_threshold: float = 0.8):
        super().__init__(test_cases)
        self.trace_file_path = trace_file_path
        self.similarity_threshold = similarity_threshold
        # 预加载所有trace结果
        self.trace_results = self._load_trace_results()
        # 加载BGE中文词嵌入模型（优先本地缓存）
        self._load_embedding_model()

    def _load_embedding_model(self):
        """加载BGE中文词嵌入模型"""
        model_name = "BAAI/bge-large-zh"
        
        try:
            self.embedding_model = SentenceTransformer(model_name,local_files_only=True)
        except Exception as e:
            self.embedding_model = SentenceTransformer(model_name)

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        if self.embedding_model is None or not text1 or not text2:
            # 降级使用字符串包含匹配
            return 1.0 if str(text2) in str(text1) else 0.0

        # 计算余弦相似度
        embedding1 = self.embedding_model.encode(str(text1), convert_to_tensor=True)
        embedding2 = self.embedding_model.encode(str(text2), convert_to_tensor=True)
        cos_sim = util.cos_sim(embedding1, embedding2)
        return float(cos_sim[0][0])

    def _load_trace_results(self) -> Dict[str, Any]:
        """加载全链路trace结果，返回case_id到完整trace数据的映射"""
        trace_map = {}
        try:
            with open(self.trace_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="加载语义升维trace结果"):
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

    def parse_expected_fields(self, case: TestCase) -> Dict[str, Any]:
        """直接从TestCase字段读取预期值，不再解析expected_cards字符串"""
        expected_fields = {}

        # 通用字段
        
        expected_fields['title'] = case.expected_title

        expected_fields['message_role'] = case.expected_message_role
    
        expected_fields['summary'] = case.expected_summary
    
        expected_fields['problem'] = case.expected_problem
    
        expected_fields['suggestion'] = case.expected_suggestion
    
        expected_fields['participants'] = case.expected_participants
    
        expected_fields['times'] = case.expected_times
    
        expected_fields['locations'] = case.expected_locations
    
        expected_fields['tags'] = case.expected_tags
    
        expected_fields['missing_fields'] = case.expected_missing_fields

        # 从decision_signals中提取is_question
        
        has_question = case.expected_decision_signals.get('has_question', 0.0) >= 0.5
        expected_fields['is_question'] = has_question

        return expected_fields

    def calculate_field_accuracy(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """计算单条用例的字段准确率，使用语义相似度匹配"""
        if not expected:
            return 1.0  # 无预期字段时默认正确
        correct_fields = 0
        total_fields = len(expected)

        for field, expected_value in expected.items():
            actual_value = actual.get(field)
            if actual_value is None:
                continue

            if isinstance(expected_value, list):
                # 列表类型字段匹配（标签/人名等，语义相似度匹配）
                # 两个列表的语义相似度（计算两两最大相似度的平均）
                similarities = []
                for exp_item in expected_value:
                    max_sim = 0.0
                    for act_item in actual_value:
                        sim = self._semantic_similarity(str(act_item), str(exp_item))
                        if sim > max_sim:
                            max_sim = sim
                    similarities.append(max_sim)
                avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                if avg_sim >= self.similarity_threshold:
                    correct_fields += 1
            elif isinstance(expected_value, str):
                # message_role字段要求严格相等，其他字符串字段使用语义相似度匹配
                if field == 'message_role':
                    if str(actual_value).strip() == str(expected_value).strip():
                        correct_fields += 1
                else:
                    # 其他字符串类型字段语义相似度匹配
                    sim = self._semantic_similarity(str(actual_value), str(expected_value))
                    if sim >= self.similarity_threshold:
                        correct_fields += 1
            elif isinstance(expected_value, bool):
                # 布尔类型精确匹配
                if actual_value == expected_value:
                    correct_fields += 1
            elif actual_value == expected_value:
                # 其他类型精确匹配
                correct_fields += 1

        return correct_fields / total_fields

    def run(self) -> Dict[str, Any]:
        total_accuracy = 0.0
        field_level_correct = {}
        field_level_total = {}
        missing_cases = 0

        for case in tqdm(self.test_cases, desc="评估语义升维模块"):
            # 从trace结果中读取实际结果
            trace_data = self.trace_results.get(case.case_id)
            if trace_data is None or 'lift' not in trace_data or trace_data['lift'] is None or not trace_data['lift'].get('cards'):
                missing_cases += 1
                continue

            # 解析实际字段
            lift_result = trace_data['lift']
            card = lift_result['cards'][0] if lift_result.get('cards') else {}
            actual_fields = {
                'title': card.get('title'),
                'message_role': card.get('message_role'),
                'summary': card.get('summary'),
                'tags': card.get('tags', []),
                'participants': card.get('participants', []),
                'times': card.get('times'),
                'problem': card.get('problem'),
                'suggestion': card.get('suggestion'),
                'locations': card.get('locations'),
                'missing_fields': card.get('missing_fields', []),
                'is_question': card.get('decision_signals', {}).get('has_question', 0) >= 0.5,       
            }
            expected_fields = self.parse_expected_fields(case)

            # 跳过无预期字段的用例
            if not expected_fields:
                continue

            # 计算字段准确率
            case_accuracy = self.calculate_field_accuracy(actual_fields, expected_fields)
            total_accuracy += case_accuracy

            # 统计字段级别准确率（使用与整体计算相同的严格规则）
            for field in expected_fields:
                field_level_total[field] = field_level_total.get(field, 0) + 1
                actual_value = actual_fields.get(field)
                expected_value = expected_fields[field]
                field_correct = False

                if actual_value is None:
                    continue

                if isinstance(expected_value, list):
                    # 列表类型字段匹配（语义相似度匹配）
                    if isinstance(actual_value, list):
                        similarities = []
                        for exp_item in expected_value:
                            max_sim = 0.0
                            for act_item in actual_value:
                                sim = self._semantic_similarity(str(act_item), str(exp_item))
                                if sim > max_sim:
                                    max_sim = sim
                            similarities.append(max_sim)
                        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                        if avg_sim >= self.similarity_threshold:
                            field_correct = True
                elif isinstance(expected_value, str):
                    # message_role字段要求严格相等，其他字符串字段使用语义相似度匹配
                    if field == 'message_role':
                        if str(actual_value).strip() == str(expected_value).strip():
                            field_correct = True
                    else:
                        # 其他字符串类型字段语义相似度匹配
                        sim = self._semantic_similarity(str(actual_value), str(expected_value))
                        if sim >= self.similarity_threshold:
                            field_correct = True
                elif isinstance(expected_value, bool):
                    # 布尔类型精确匹配
                    if actual_value == expected_value:
                        field_correct = True
                elif actual_value == expected_value:
                    # 其他类型精确匹配
                    field_correct = True

                if field_correct:
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
        valid_cases = len(self.results)
        avg_accuracy = total_accuracy / valid_cases if valid_cases > 0 else 0
        field_accuracies = {
            field: round(field_level_correct.get(field, 0) / field_level_total.get(field, 1), 4)
            for field in field_level_total
        }

        result = {
            "score": self.calculate_score(avg_accuracy),
            "total_cases": len(self.test_cases),
            "valid_cases": valid_cases,
            "missing_cases": missing_cases,
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
    from eval_utils import load_eval_cases
    test_cases = load_eval_cases()
    evaluator = SemanticLifterEvaluator(test_cases)
    result = evaluator.run()
    print(f"语义升维模块得分: {result['score']:.2f}/100")
    print(f"总用例数: {result['total_cases']}, 有效用例: {result['valid_cases']}, 缺失用例: {result['missing_cases']}")
    print(f"平均字段提取准确率: {result['average_accuracy']:.4f}")
    print("字段级别准确率:", result['field_level_accuracies'])
