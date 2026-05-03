"""
RAG模块评估：检索召回率、Top-k准确率、答案相关性、幻觉率
"""
from typing import List, Dict, Any, Tuple
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from insight.core import try_answer_with_rag, save_knowledge
from insight.core.lift import lift_candidates
from insight.core.detect import DetectionResult

class RAGEvaluator(BaseModuleEvaluator):
    def __init__(self, test_cases: List[TestCase]):
        super().__init__(test_cases)
        # 准备测试知识库（从测试用例中的知识场景构建）
        self._build_test_knowledge_base()

    def _build_test_knowledge_base(self):
        """构建测试知识库"""
        knowledge_cases = [c for c in self.test_cases if c.expected_target_pool == 'knowledge']
        for case in knowledge_cases:
            detect_result = DetectionResult(
                messages=case.conversation + [case.trigger_message],
                value_score=(case.expected_detect_score_min + case.expected_detect_score_max) / 2
            )
            lift_result = lift_candidates(detect_result)
            if lift_result.cards:
                # 保存知识到知识库
                save_knowledge(lift_result.cards[0])

    def run(self) -> Dict[str, Any]:
        total_recall = 0.0
        total_top1_accuracy = 0.0
        total_top3_accuracy = 0.0
        total_relevance = 0.0
        total_hallucination_count = 0
        valid_cases = 0

        for case in self.test_cases:
            # 只处理需要RAG的用例（观察问答场景、知识检索场景）
            if not ('RAG' in case.expected_path or 'rag' in case.expected_path.lower()):
                continue

            # 构造查询
            query = case.trigger_message
            expected_knowledge_ids = []
            # 从预期中提取应该命中的知识ID
            if case.expected_target_pool == 'knowledge':
                expected_knowledge_ids.append(case.case_id)
            elif '命中RAG' in case.scenario:
                # 问答场景需要匹配对应的知识
                expected_knowledge_ids = [c.case_id for c in self.test_cases if c.expected_target_pool == 'knowledge' and c.case_id.split('-')[1] == case.case_id.split('-')[1]]

            # 调用RAG检索
            results = self.retriever.retrieve(query, top_k=5)
            retrieved_ids = [res.get('metadata', {}).get('case_id') for res in results]
            retrieved_scores = [res.get('score', 0) for res in results]

            # 1. 计算召回率
            hits = len(set(expected_knowledge_ids) & set(retrieved_ids))
            recall = hits / len(expected_knowledge_ids) if expected_knowledge_ids else 1.0
            total_recall += recall

            # 2. 计算Top-k准确率
            top1_accuracy = 1.0 if expected_knowledge_ids and retrieved_ids and retrieved_ids[0] in expected_knowledge_ids else 0.0
            top3_accuracy = 1.0 if expected_knowledge_ids and set(expected_knowledge_ids) & set(retrieved_ids[:3]) else 0.0
            total_top1_accuracy += top1_accuracy
            total_top3_accuracy += top3_accuracy

            # 3. 计算相关性（平均得分）
            relevance = 0.0
            for res in results:
                if res.get('metadata', {}).get('case_id') in expected_knowledge_ids:
                    relevance += res.get('score', 0)
            relevance = relevance / len(expected_knowledge_ids) if expected_knowledge_ids else 0.0
            total_relevance += relevance

            # 4. 计算幻觉率（检索结果中不相关的比例）
            irrelevant_count = len([rid for rid in retrieved_ids if rid not in expected_knowledge_ids])
            hallucination_rate = irrelevant_count / len(retrieved_ids) if retrieved_ids else 0.0
            total_hallucination_count += hallucination_rate

            valid_cases += 1

            # 保存详细结果
            self.results.append({
                "case_id": case.case_id,
                "scenario": case.scenario,
                "query": query,
                "expected_knowledge_ids": expected_knowledge_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_scores": [round(s, 4) for s in retrieved_scores],
                "recall": round(recall, 4),
                "top1_accuracy": round(top1_accuracy, 4),
                "top3_accuracy": round(top3_accuracy, 4),
                "relevance": round(relevance, 4),
                "hallucination_rate": round(hallucination_rate, 4)
            })

        # 计算整体指标
        avg_recall = total_recall / valid_cases if valid_cases > 0 else 0
        avg_top1_accuracy = total_top1_accuracy / valid_cases if valid_cases > 0 else 0
        avg_top3_accuracy = total_top3_accuracy / valid_cases if valid_cases > 0 else 0
        avg_relevance = total_relevance / valid_cases if valid_cases > 0 else 0
        avg_hallucination_rate = total_hallucination_count / valid_cases if valid_cases > 0 else 0

        result = {
            "score": self.calculate_score(avg_recall, avg_top3_accuracy, avg_relevance, avg_hallucination_rate),
            "total_cases": valid_cases,
            "average_recall": round(avg_recall, 4),
            "average_top1_accuracy": round(avg_top1_accuracy, 4),
            "average_top3_accuracy": round(avg_top3_accuracy, 4),
            "average_relevance": round(avg_relevance, 4),
            "average_hallucination_rate": round(avg_hallucination_rate, 4),
            "details": self.results
        }

        save_metric_result("rag", result)
        return result

    def calculate_score(self, recall: float, top3_accuracy: float, relevance: float, hallucination_rate: float) -> float:
        """
        计算RAG模块得分：
        - 召回率占30%
        - Top3准确率占30%
        - 相关性占20%
        - 幻觉率扣分项（幻觉率越低得分越高）占20%
        """
        score = (recall * 0.3 + top3_accuracy * 0.3 + relevance * 0.2 + (1 - hallucination_rate) * 0.2) * 100
        return max(0, min(100, score))

if __name__ == "__main__":
    from utils import load_test_cases
    test_cases = load_test_cases()
    evaluator = RAGEvaluator(test_cases)
    result = evaluator.run()
    print(f"RAG模块得分: {result['score']:.2f}/100")
    print(f"平均召回率: {result['average_recall']:.4f}")
    print(f"Top1准确率: {result['average_top1_accuracy']:.4f}, Top3准确率: {result['average_top3_accuracy']:.4f}")
    print(f"平均相关性: {result['average_relevance']:.4f}, 平均幻觉率: {result['average_hallucination_rate']:.4f}")