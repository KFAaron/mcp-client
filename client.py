import asyncio
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from mcpserverconnector import MCPServerConnector

load_dotenv()

class MCPClient:
    """
    MCPClient 负责与 DeepSeek 及 MCPServerConnector 交互，实现自然语言到高德地图服务的桥接。
    """
    def __init__(self):
        self.connector = None
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in environment variables.")
        self.deepseekClient = OpenAI(api_key=api_key, base_url='https://api.deepseek.com')
        print("[LOG] MCPClient 初始化完成。")

    def build_tools_for_deepseek(self):
        """
        将 MCP 工具列表转换为 DeepSeek 可用的工具格式。
        """
        tools = []
        for tool in self.connector.tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            })
        # print(f"[LOG] 构建 DeepSeek 工具列表，共 {len(tools)} 个工具。")
        return tools

    async def call_tool(self, tool_name: str, tool_args: dict):
        """
        调用 MCP 工具，兼容 JSONRPCMessage 解析异常。
        """
        if not self.connector or not self.connector.session:
            raise RuntimeError("MCP session not initialized.")
        print(f"[LOG] 调用 MCP 工具: {tool_name}, 参数: {tool_args}")
        try:
            return await self.connector.session.call_tool(tool_name, tool_args)
        except Exception as e:
            print(f"[ERROR] 工具调用异常: {e}")
            # 兼容 JSONRPCMessage 解析异常，返回伪tool_result
            return type('ToolResult', (), {"content": f"[工具调用异常] {e}"})()

    async def process_query(self, query: str) -> str:
        """
        处理用户 query，调用 DeepSeek 和 MCP 工具。
        """
        messages = [{"role": "user", "content": query}]
        tools = self.build_tools_for_deepseek()

        # print(f"[LOG] 向 DeepSeek 发送用户问题: {query}")
        response = self.deepseekClient.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )

        final_text = []
        while True:
            choice = response.choices[0]
            message = choice.message
            if not getattr(message, "tool_calls", None):
                # 没有工具调用，直接输出
                final_text.append(message.content)
                # print(f"[LOG] DeepSeek 回复:\n {message.content}")
                break

            # 有工具调用，先插入assistant消息（带tool_calls字段）
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [tc.model_dump() for tc in message.tool_calls]
            })
            # print(f"[LOG] {message.content}")
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                # 兼容多种参数类型
                if hasattr(arguments, 'model_dump'):
                    tool_args = arguments.model_dump()
                elif isinstance(arguments, str):
                    tool_args = json.loads(arguments)
                else:
                    tool_args = arguments
                tool_name = tool_call.function.name
                tool_result = await self.call_tool(tool_name, tool_args)
                # print(f"[LOG] [调用工具 {tool_name} with args {tool_args}]")
                # final_text.append(f"[调用工具 {tool_name} with args {tool_args}]")

                # 构造 tool result 消息，role 必须为 tool
                content = tool_result.content
                if hasattr(content, 'model_dump'):
                    content = content.model_dump()
                elif not isinstance(content, (str, dict)):
                    content = str(content)
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })
                print(f"[LOG] 工具 {tool_name} 返回: {content}")

            # 再次调用大模型，获得最终回复
            # print(f"[LOG] 向 DeepSeek 发送工具调用结果，等待新回复...")
            response = self.deepseekClient.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools
            )

        return "\n".join(final_text)

    async def chat_loop(self):
        """
        执行一个交互聊天循环。
        """
        print("[LOG] MCP客户端开始！")
        print("[LOG] 输入你的问题或者输'quit'退出")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    print("[LOG] 用户主动退出。"); break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\n报错了: {str(e)}")
                print(f"[LOG] 异常详情: {e}")

async def main():
    """
    主入口，负责初始化客户端并启动聊天循环。
    """
    client = MCPClient()
    try:
        async with MCPServerConnector() as connector:
            client.connector = connector
            await client.chat_loop()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("[LOG] 用户中断，正在优雅退出...")
    finally:
        print("结束了")

if __name__ == "__main__":
    asyncio.run(main())



