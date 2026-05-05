"""
通用评估工具和数据加载模块
"""
import csv
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class TestCase:
    """测试用例数据结构"""
    case_id: str
    chat_type: str
    scenario: str
    priority: str
    coverage_path: str
    preconditions: str
    conversation: List[str]
    round_count: int
    expected_detect_score_min: int
    expected_detect_score_max: int
    expected_path: str
    expected_cards: str
    expected_target_pool: str
    # 新增语义升维预期字段
    expected_title: str
    expected_message_role: str
    expected_summary: str
    expected_problem: str
    expected_suggestion: str
    expected_participants: List[str]
    expected_times: str
    expected_locations: str
    expected_tags: List[str]
    expected_decision_signals: Dict[str, Any]
    expected_missing_fields: List[str]
    # 原有字段
    status_assertions: str
    engine_result_assertions: str
    failure_assertions: str
    automation_assertions: str
    remark: str

def load_eval_cases(csv_path: str = "D:\\MasterDegreeCandidate\\Projects\\Feishu Competition\\So-Free-Knowledge\\datas\\chat_test_2.csv") -> List[TestCase]:
    """加载评估用例"""
    test_cases = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 解析对话历史，去除前面的序号前缀（如 "01. "、"12. " 等）
            import re
            conversation = []
            for line in row['连续对话（至少10轮）'].split('\n'):
                line = line.strip()
                if not line:
                    continue           
                conversation.append(line)

            # 解析预期detect_score范围
            score_str = row['预期detect_score'].strip()
            if score_str == 'N/A' or not score_str:
                min_score = 0
                max_score = 100
            else:
                # 清洗字符串，只保留数字、点号和分隔符
                import re
                score_str_clean = re.sub(r'[^\d.-]', '-', score_str)
                score_range = [s for s in score_str_clean.split('-') if s.strip()]
                min_score = int(float(score_range[0])) if len(score_range) > 0 else 0
                max_score = int(float(score_range[1])) if len(score_range) > 1 else min_score if len(score_range) > 0 else 100

            # 解析列表类型字段
            expected_participants = eval(row.get('预期participants', '[]').strip())
            
            expected_tags = eval(row.get('预期tags', '[]').strip())
            
            expected_missing_fields = eval(row.get('预期missing_fields', '[]').strip())
        
            # 解析JSON类型字段
            expected_decision_signals = {}
            if row.get('预期decision_signals_json', '').strip():
                try:
                    expected_decision_signals = json.loads(row['预期decision_signals_json'])
                except:
                    pass

            test_case = TestCase(
                case_id=row['用例ID'],
                chat_type=row['聊天类型'],
                scenario=row['场景名称'],
                priority=row['优先级'],
                coverage_path=row['覆盖节点/分支'],
                preconditions=row['前置条件'],
                conversation=conversation,
                round_count=int(row['对话轮数']),
                expected_detect_score_min=min_score,
                expected_detect_score_max=max_score,
                expected_path=row['预期图路径'],
                expected_cards=row['预期cards字段'],
                expected_target_pool=row['预期decisions/target_pool'],
                # 新增语义升维预期字段
                expected_title=row.get('预期title', '').strip(),
                expected_message_role=row.get('预期message_role', '').strip(),
                expected_summary=row.get('预期summary', '').strip(),
                expected_problem=row.get('预期problem', '').strip(),
                expected_suggestion=row.get('预期suggestion', '').strip(),
                expected_participants=expected_participants,
                expected_times=row.get('预期times', '').strip(),
                expected_locations=row.get('预期locations', '').strip(),
                expected_tags=expected_tags,
                expected_decision_signals=expected_decision_signals,
                expected_missing_fields=expected_missing_fields,
                # 原有字段
                status_assertions=row['状态字段断言'],
                engine_result_assertions=row['EngineResult断言'],
                failure_assertions=row['失败/告警断言'],
                automation_assertions=row['自动化断言建议'],
                remark=row['备注']
            )
            test_cases.append(test_case)
    return test_cases

def save_metric_result(metric_name: str, result: Dict[str, Any], output_dir: str = "outputs/insight_module_eval"):
    """保存指标结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{metric_name}_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"指标 {metric_name} 结果已保存到 {output_path}")

def generate_summary_report(all_results: Dict[str, Any], output_dir: str = "outputs/insight_module_eval"):
    """生成汇总测试报告"""
    summary = {
        "test_time": os.popen('date +"%Y-%m-%d %H:%M:%S"').read().strip(),
        "total_test_cases": len(load_eval_cases()),
        "module_results": all_results,
        "overall_score": sum(res.get('score', 0) for res in all_results.values()) / len(all_results) if all_results else 0
    }

    output_path = os.path.join(output_dir, "summary_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Insight模块评估汇总报告\n\n")
        f.write(f"测试时间: {summary['test_time']}\n")
        f.write(f"总测试用例数: {summary['total_test_cases']}\n")
        f.write(f"总体得分: {summary['overall_score']:.2f}/100\n\n")

        f.write("## 各模块评估结果\n\n")
        for module, result in all_results.items():
            f.write(f"### {module}\n")
            f.write(f"- 得分: {result.get('score', 0):.2f}/100\n")
            for metric, value in result.items():
                if metric != 'score' and metric != 'details':
                    f.write(f"- {metric}: {value}\n")
            f.write("\n")

    print(f"汇总报告已保存到 {output_path}")

class BaseModuleEvaluator:
    """模块评估基类"""
    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
        self.results = []

    def run(self) -> Dict[str, Any]:
        """运行测试，返回指标结果"""
        raise NotImplementedError("子类必须实现run方法")

    def calculate_score(self) -> float:
        """计算模块得分（0-100）"""
        raise NotImplementedError("子类必须实现calculate_score方法")