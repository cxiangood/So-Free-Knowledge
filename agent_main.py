#!/usr/bin/env python3
"""
So-Free-Knowledge Harness Agent 入口文件
支持两种运行模式：
1. 服务模式：启动API服务，接收HTTP调用
2. 命令行模式：直接通过命令行调用Agent能力
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from agent.interface import SoFreeKnowledgeAgent
from utils import configure_logging


def run_service_mode(args):
    """运行API服务模式"""
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        print("Error: Service mode requires fastapi and uvicorn. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    # 初始化Agent
    agent = SoFreeKnowledgeAgent(
        state_dir=args.state_dir,
        env_file=args.env_file,
        rag_enabled=not args.no_rag,
        task_push_enabled=args.task_push_enabled,
        task_push_chat_id=args.task_push_chat_id,
    )

    app = FastAPI(title="So-Free-Knowledge Agent API", version="1.0.0")

    class InvokeRequest(BaseModel):
        input: Any
        context: Dict[str, Any] = None
        config: Dict[str, Any] = None

    class ToolCallRequest(BaseModel):
        name: str
        parameters: Dict[str, Any]

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "agent": agent.name, "description": agent.description}

    @app.get("/tools")
    async def list_tools(category: str = None):
        """列出所有可用工具"""
        tools = agent.tool_registry.list_tools(category)
        return {
            "count": len(tools),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "parameters": t.input_schema.model_json_schema()
                }
                for t in tools
            ]
        }

    @app.post("/invoke")
    async def invoke_agent(request: InvokeRequest):
        """调用Agent"""
        try:
            from agent.interface import AgentContext
            context = AgentContext(session_id=request.context.get("session_id", "")) if request.context else None
            response = agent.invoke(request.input, context, request.config)
            return {
                "content": response.content,
                "tool_calls": [tc.__dict__ for tc in response.tool_calls],
                "tool_results": [tr.__dict__ for tr in response.tool_results],
                "metadata": response.metadata
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/tools/{tool_name}/invoke")
    async def invoke_tool(tool_name: str, request: ToolCallRequest):
        """直接调用工具"""
        try:
            result = agent.tool_registry.invoke_tool(tool_name, request.parameters)
            if not result.get("success", False):
                raise HTTPException(status_code=400, detail=result.get("error", "Tool execution failed"))
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    print(f"Starting So-Free-Knowledge Agent service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def run_cli_mode(args):
    """运行命令行模式"""
    # 初始化Agent
    agent = SoFreeKnowledgeAgent(
        state_dir=args.state_dir,
        env_file=args.env_file,
        rag_enabled=not args.no_rag,
        task_push_enabled=args.task_push_enabled,
        task_push_chat_id=args.task_push_chat_id,
    )

    if args.list_tools:
        # 列出所有工具
        tools = agent.tool_registry.list_tools(args.category)
        print(f"Available tools ({len(tools)}):")
        print("-" * 80)
        for t in tools:
            print(f"\033[1m{t.name}\033[0m [{t.category}]")
            print(f"  {t.description}")
            print()
        return

    if args.tool_name:
        # 调用指定工具
        try:
            parameters = json.loads(args.parameters) if args.parameters else {}
            result = agent.tool_registry.invoke_tool(args.tool_name, parameters)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.query:
        # 自然语言查询
        response = agent.invoke(args.query)
        print(f"\033[1m回答:\033[0m {response.content}")
        if response.metadata:
            print(f"\n\033[1m元数据:\033[0m")
            print(json.dumps(response.metadata, ensure_ascii=False, indent=2))
        return

    # 交互式模式
    print("So-Free-Knowledge Agent 交互式模式")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 80)

    from agent.interface import AgentContext
    context = AgentContext(session_id="cli_session")

    while True:
        try:
            query = input("\n> ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            response = agent.invoke(query, context)
            print(f"\n{response.content}")

            if response.metadata and response.metadata.get("success") and response.metadata.get("sources"):
                print(f"\n参考来源 ({len(response.metadata['sources'])}):")
                for i, source in enumerate(response.metadata["sources"], 1):
                    print(f"  {i}. {source.get('title', '无标题')} (score: {source.get('score', 0):.3f})")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


def main():
    configure_logging(app_name="SOFREE_AGENT", quiet=False)

    parser = argparse.ArgumentParser(description="So-Free-Knowledge Harness Agent")
    parser.add_argument("--mode", choices=["service", "cli"], default="cli", help="运行模式")
    parser.add_argument("--state-dir", default="outputs/agent_state", help="状态存储目录")
    parser.add_argument("--env-file", default=".env", help="环境变量文件路径")
    parser.add_argument("--no-rag", action="store_true", help="禁用RAG功能")
    parser.add_argument("--task-push-enabled", action="store_true", help="启用任务推送")
    parser.add_argument("--task-push-chat-id", default="", help="任务推送的聊天ID")

    # 服务模式参数
    service_group = parser.add_argument_group("Service mode options")
    service_group.add_argument("--host", default="0.0.0.0", help="服务监听地址")
    service_group.add_argument("--port", type=int, default=8000, help="服务监听端口")

    # CLI模式参数
    cli_group = parser.add_argument_group("CLI mode options")
    cli_group.add_argument("--list-tools", action="store_true", help="列出所有可用工具")
    cli_group.add_argument("--category", help="工具类别过滤")
    cli_group.add_argument("--tool-name", help="要调用的工具名称")
    cli_group.add_argument("--parameters", help="工具参数，JSON格式")
    cli_group.add_argument("--query", help="自然语言查询")

    args = parser.parse_args()

    if args.mode == "service":
        run_service_mode(args)
    else:
        run_cli_mode(args)


if __name__ == "__main__":
    main()
