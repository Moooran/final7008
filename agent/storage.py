# agent/storage.py
import json
import os
from datetime import datetime
from typing import Any, Dict

from .config import AgentConfig

class Storage:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def save_screenshot(self, image_bytes: bytes, step_id: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.cfg.log_dir, f"{ts}_{step_id}.png")
        with open(path, "wb") as f:
            f.write(image_bytes)
        return path

    def save_dom_summary(self, summary: Dict[str, Any], step_id: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.cfg.log_dir, f"{ts}_{step_id}_dom.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return path

    def save_final_output(self, result: Dict[str, Any]) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.cfg.output_dir, f"{ts}_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return path