# agent/tools/ocr_tool.py
from typing import Optional

class OcrTool:
    async def ocr_image(self, image_bytes: bytes) -> str:
        # TODO: 集成 paddleocr 或 Tesseract
        return "mock ocr text"