#!/usr/bin/env python3
"""测试LLM客户端连接"""

import sys

from llm.client import LLMConfig, LLMClient


def test_llm_connection():
    """测试LLM连接"""
    print("正在加载LLM配置...")

    # 从环境变量加载配置
    config = LLMConfig.from_env()

    # 检查配置是否完整
    missing_fields = config.missing_fields()
    if missing_fields:
        print(f"配置不完整，缺少: {', '.join(missing_fields)}")
        print("请确保以下环境变量已设置:")
        print("- LLM_API_KEY: 你的API密钥")
        print("- LLM_MODEL_ID: 模型ID")
        print("- LLM_BASE_URL: API基础URL (例如: https://ark.cn-beijing.volces.com/api/v3)")
        return False

    print(f"配置加载成功:")
    print(f"  模型ID: {config.model_id}")
    print(f"  基础URL: {config.base_url}")
    print(f"  temperature: {config.temperature}")
    print(f"  max_tokens: {config.max_tokens}")

    # 创建客户端
    client = LLMClient(config)

    # 测试消息
    system_prompt = "你是一个有用的助手，请用简洁的语言回答问题。"
    user_message = "请简单介绍一下你自己，用不超过30个字。"

    print("\n正在发送测试请求...")
    print(f"System Prompt: {system_prompt}")
    print(f"User Message: {user_message}")
    print("-" * 50)

    # 调用LLM
    response = client.build_reply(system_prompt, user_message)

    print(f"Response: {response}")
    print("-" * 50)

    if "LLM 调用失败" in response or "配置不完整" in response:
        print("❌ 测试失败")
        return False
    else:
        print("✅ 测试成功！LLM连接正常。")
        return True


if __name__ == "__main__":
    test_llm_connection()
