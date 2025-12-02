import asyncio
from typing import Any

from .controller import AgentController
from .config import AgentConfig
from .planner import Planner


class WebAgent:
    """同步封装，用于 main.py 中调用。

    将 models/ 下的 VLLM 模型实例注入到 Planner 里，
    然后使用 AgentController 跑完整个状态机。
    """

    def __init__(self, vllm_model: Any, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()
        # 创建带具体模型的 Planner，并注入到控制器
        planner = Planner(vllm_model)
        self.controller = AgentController(self.cfg, planner=planner)

    def run(self, task: str, max_steps: int | None = None) -> dict:
        """同步入口，内部启动 asyncio 运行控制器。"""

        async def _run():
            return await self.controller.run(task, max_steps=max_steps)

        return asyncio.run(_run())
