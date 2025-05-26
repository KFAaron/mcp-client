import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession

class MCPServerConnector:
    """
    MCPServerConnector 负责与 MCP Server 建立 SSE 连接并管理会话。
    支持 async with 用法，确保资源自动释放。
    """
    def __init__(self):
        self.session = None
        self.tools = {}
        self._sse_cm = None
        self._sse_streams = None
        self._session_cm = None

    async def __aenter__(self):
        """
        进入异步上下文，建立 SSE 连接并初始化会话。
        """
        # 读取配置文件，获取 MCP Server 地址
        with open('mcp.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        url = config["mcpServers"]["amap-amap-sse"]["url"]
        print(f"[LOG] 尝试连接到 MCP Server: {url}")
        self._sse_cm = sse_client(url)
        self._sse_streams = await self._sse_cm.__aenter__()
        print("[LOG] SSE 流已获取。")
        self._session_cm = ClientSession(self._sse_streams[0], self._sse_streams[1])
        self.session = await self._session_cm.__aenter__()
        print("[LOG] ClientSession 已创建。")
        await self.session.initialize()
        print("[LOG] Session 已初始化。")
        response = await self.session.list_tools()
        self.tools = {tool.name: tool for tool in response.tools}
        print(f"[LOG] 成功获取 {len(self.tools)} 个工具。")
        print("[LOG] 连接成功并准备就绪。")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        退出异步上下文，关闭会话和 SSE 连接。
        """
        print("[LOG] 正在关闭 MCPServerConnector 资源...")
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc, tb)
            print("[LOG] ClientSession 已关闭。")
        if self._sse_cm:
            await self._sse_cm.__aexit__(exc_type, exc, tb)
            print("[LOG] SSE 连接已关闭。")



