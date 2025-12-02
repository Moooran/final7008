# agent/planner.py
from dataclasses import dataclass
from typing import Any, Dict, Literal

ActionType = Literal["goto", "click", "type", "scroll", "press",
                     "download", "final_answer", "retry", "error"]


@dataclass
class AgentAction:
    type: ActionType
    params: Dict[str, Any]


@dataclass
class PlannerInput:
    task: str
    history: list[Dict[str, Any]]
    dom_summary: Dict[str, Any] | None = None
    screenshot_path: str | None = None
    current_url: str | None = None


class Planner:
    """调用 VLLM 模型，根据页面信息和任务决定下一步动作。"""

    def __init__(self, model: Any):
        # model: models/ 下的任意 *VLLM 实例，需实现 analyze()
        self.model = model

    async def plan_next(self, inp: PlannerInput) -> AgentAction:
        """PLAN / ANALYSIS 阶段：调用大模型返回动作 JSON。"""

        # 调用统一的 analyze 接口（见 models/base_model.py / generic_model.py）
        model_result = self.model.analyze(
            screenshot_path=inp.screenshot_path,
            dom_summary=str(inp.dom_summary or ""),
            user_goal=inp.task,
            current_url=inp.current_url or "",
        )
        action_str = str(model_result.get("action", "")).upper().strip()
        param = str(model_result.get("parameter", "")).strip()
        completed = bool(model_result.get("completed", False))

        valid_actions = {"GOTO", "CLICK", "TYPE", "SCROLL", "DONE"}

        # 第一轮禁止 DONE：必须先 GOTO 打开网站
        is_first_step = len(inp.history) == 0
        if is_first_step and action_str == "DONE":
            # 如果 parameter 是 URL，就强制改为 GOTO 该 URL
            if param.startswith("http://") or param.startswith("https://"):
                action_str = "GOTO"
            else:
                return AgentAction(
                    type="final_answer",
                    params={"answer": model_result, "error": "first_step_done_without_url"},
                )

        if action_str not in valid_actions:
            # 模型输出了未知动作，直接结束，避免 FSM 收到垃圾动作
            return AgentAction(type="final_answer", params={"answer": model_result, "error": "invalid_action"})

        if completed or action_str == "DONE":
            return AgentAction(type="final_answer", params={"answer": model_result})

        if action_str == "GOTO":
            # 简单 URL 校验
            if not (param.startswith("http://") or param.startswith("https://")):
                return AgentAction(type="final_answer", params={"answer": model_result, "error": "invalid_url"})
            return AgentAction(type="goto", params={"url": param})

        if action_str == "CLICK":
            if not param:
                return AgentAction(type="final_answer", params={"answer": model_result, "error": "empty_selector"})
            return AgentAction(type="click", params={"selector": param})

        if action_str == "TYPE":
            # parameter 必须是 selector:::text，缺省时使用 input 作为 selector
            if ":::" in param:
                sel, txt = param.split(":::", 1)
            else:
                sel, txt = "input", param
            return AgentAction(type="type", params={"selector": sel, "text": txt})

        if action_str == "SCROLL":
            direction = param.lower() if param.lower() in {"up", "down"} else "down"
            return AgentAction(type="scroll", params={"direction": direction})

        # 理论上走不到这里
        return AgentAction(type="final_answer", params={"answer": model_result, "error": "unreachable"})
