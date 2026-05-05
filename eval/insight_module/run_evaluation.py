"""
Insight模块评估主运行脚本
"""
import sys
import os
# 先添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 再添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_utils import load_eval_cases, generate_summary_report
from eval_signal_detector import SignalDetectorEvaluator
from eval_semantic_lifter import SemanticLifterEvaluator
from eval_router import RouterEvaluator
from eval_observe import ObserveEvaluator
from eval_rag import RAGEvaluator

# 全链路trace结果文件路径（与run_full_pipeline_trace.py输出保持一致）
TRACE_FILE_PATH = "outputs/insight_module_eval/insight_full_pipeline_chat_test_2.jsonl"

def main():
    print("=" * 60)
    print("开始Insight模块级评估")
    print("=" * 60)

    # 加载评估用例
    print("\n1. 加载评估用例...")
    eval_cases = load_eval_cases()
    print(f"   共加载 {len(eval_cases)} 条评估用例")

    # 运行各模块评估
    evaluators = [
        ("信号检测模块", SignalDetectorEvaluator),
        ("语义升维模块", SemanticLifterEvaluator),
        ("路由模块", RouterEvaluator),
        ("观察池模块", ObserveEvaluator),
        ("RAG模块", RAGEvaluator)
    ]

    all_results = {}

    for module_name, evaluator_class in evaluators:
        print(f"\n2. 评估{module_name}...")
        evaluator = evaluator_class(eval_cases, trace_file_path=TRACE_FILE_PATH)
        result = evaluator.run()
        all_results[module_name] = result
        print(f"   {module_name}得分: {result['score']:.2f}/100")
        for key, value in result.items():
            if key not in ['score', 'details']:
                print(f"   - {key}: {value}")

    # 生成汇总报告
    print("\n3. 生成汇总评估报告...")
    generate_summary_report(all_results)

    print("\n" + "=" * 60)
    print("Insight模块级评估完成！")
    print("评估结果已保存到 outputs/insight_module_eval/ 目录")
    print("=" * 60)

if __name__ == "__main__":
    main()