# Insight模块评估套件

## 目录结构
```
eval/insight_module/
├── utils.py                  # 通用评估工具和数据加载
├── eval_signal_detector.py   # 信号检测模块评估
├── eval_semantic_lifter.py   # 语义升维模块评估
├── eval_router.py            # 路由模块评估
├── eval_observe.py           # 观察池模块评估
├── eval_rag.py               # RAG模块评估
├── run_evaluation.py         # 评估主运行脚本
└── README.md                 # 说明文档

outputs/insight_module_eval/  # 评估结果输出目录
├── signal_detector_result.json
├── semantic_lifter_result.json
├── router_result.json
├── observe_result.json
├── rag_result.json
└── summary_report.md         # 汇总评估报告
```

## 评估指标说明
| 模块 | 评估指标 | 计算方式 |
|------|----------|----------|
| **信号检测模块** | 精确率(Precision)、召回率(Recall)、F1值 | 高价值信号识别的混淆矩阵计算 |
| **语义升维模块** | 字段提取准确率 | 结构化字段（任务/知识/问题字段）的提取正确率 |
| **路由模块** | 路由准确率 | 消息路由到正确下游节点的比例 |
| **观察池模块** | 问题识别准确率 | 问题类消息识别的正确率 |
| **RAG模块** | 召回率、Top-k准确率、答案相关性、幻觉率 | 检索效果和回答质量评估 |

## 运行方式
```bash
cd eval/insight_module
python run_evaluation.py
```

## 输出说明
1. 每个模块的评估结果会保存为单独的JSON文件，包含详细的每个用例的评估结果
2. 汇总报告 `summary_report.md` 包含各模块得分和整体评估结果
3. 所有结果保存在 `outputs/insight_module_eval/` 目录下

## 自定义评估
可以通过修改 `run_evaluation.py` 中的评估器列表，选择需要评估的模块。
如需添加新的评估指标，可以在对应模块的评估脚本中扩展计算逻辑。