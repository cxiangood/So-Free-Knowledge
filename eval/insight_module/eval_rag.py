"""
RAG模块评估：检索召回率、Top-k准确率、答案相关性、幻觉率
直接读取全链路trace结果文件进行评估，无需运行模块
"""
import json
from typing import List, Dict, Any
from eval_utils import TestCase, BaseModuleEvaluator, save_metric_result
from tqdm import tqdm

class RAGEvaluator(BaseModuleEvaluator):
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
                for line in tqdm(f, desc="加载RAG trace结果"):
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
        total_recall = 0.0
        total_top1_accuracy = 0.0
        total_top3_accuracy = 0.0
        total_relevance = 0.0
        total_hallucination_count = 0
        valid_cases = 0
        missing_cases = 0

        for case in tqdm(self.test_cases, desc="评估RAG模块"):
            # 只处理需要RAG的用例（观察问答场景、知识检索场景）
            if not ('RAG' in case.expected_path or 'rag' in case.expected_path.lower()):
                continue

            # 从trace结果中读取RAG相关结果
            rag_result = self.trace_results.get(case.case_id)
            if rag_result is None:
                missing_cases += 1
                continue

            # 提取检索结果
            retrieved_docs = []
            # 从obs结果中提取RAG检索结果
            if rag_result['obs'] and rag_result['obs'].get('rag_retrieval'):
                retrieved_docs = rag_result['obs']['rag_retrieval'].get('documents', [])
            # 从task结果中提取RAG增强结果
            elif rag_result['task'] and rag_result['task'].get('rag_enhancement'):
                retrieved_docs = rag_result['task']['rag_enhancement'].get('documents', [])

            # 预期命中的知识ID（从测试用例备注或预期字段中提取）
            expected_knowledge_ids = []
            if case.remark and 'knowledge_id=' in case.remark:
                expected_knowledge_ids = [kid.strip() for kid in case.remark.split('knowledge_id=')[1].split(',')]

            if not expected_knowledge_ids:
                continue

            # 1. 计算召回率
            retrieved_ids = [doc.get('metadata', {}).get('case_id', '') for doc in retrieved_docs]
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
            for doc in retrieved_docs:
                if doc.get('metadata', {}).get('case_id') in expected_knowledge_ids:
                    relevance += doc.get('score', 0)
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
                "query": case.trigger_message,
                "expected_knowledge_ids": expected_knowledge_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_scores": [round(doc.get('score', 0), 4) for doc in retrieved_docs],
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
            "total_cases": len([c for c in self.test_cases if 'RAG' in c.expected_path or 'rag' in c.expected_path.lower()]),
            "valid_cases": valid_cases,
            "missing_cases": missing_cases,
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
    from eval_utils import load_eval_cases
    test_cases = load_eval_cases()
    evaluator = RAGEvaluator(test_cases)
    result = evaluator.run()
    print(f"RAG模块得分: {result['score']:.2f}/100")
    print(f"总用例数: {result['total_cases']}, 有效用例: {result['valid_cases']}, 缺失用例: {result['missing_cases']}")
    print(f"平均召回率: {result['average_recall']:.4f}")
    print(f"Top1准确率: {result['average_top1_accuracy']:.4f}, Top3准确率: {result['average_top3_accuracy']:.4f}")
    print(f"平均相关性: {result['average_relevance']:.4f}, 平均幻觉率: {result['average_hallucination_rate']:.4f}")
