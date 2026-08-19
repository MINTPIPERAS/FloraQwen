"""FloraQwen 智能体编排层。

职责:
    - 识别: 图片 -> Qwen3-VL -> 结构化 JSON (中文名/学名/科属/置信度/依据)
    - 兜底: 低置信度时主动承认不确定, 请求补充拍摄
    - 对话: 多轮上下文 (M5 渐进实现)

用法 (在项目根目录):
    from floraqwen import FloraAgent
    agent = FloraAgent()                       # 默认 Qwen/Qwen3-VL-4B-Instruct
    result = agent.recognize("data/raw/images/rose_001.jpg")
    # result: {"ok": True, "data": {...}, "raw": "...", "note": "..."}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# 识别提示词: 要求模型输出 JSON, 便于结构化解析
IDENTIFY_PROMPT = (
    "请识别这张照片中的植物，并严格按以下 JSON 格式回答（不要输出其他内容）：\n"
    '{"中文名": "...", "学名": "...", "科属": "...", "置信度": 0.0-1.0, '
    '"识别依据": ["特征1", "特征2"], "可能混淆物种": ["..."]}\n'
    "如果无法确定，置信度请填 0.3 以下。"
)

CONFIDENCE_THRESHOLD = 0.5


class FloraAgent:
    """花草识别智能体。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        adapter: str | None = None,
        quant: str = "4bit",
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.adapter = adapter
        self.quant = quant
        self.threshold = threshold
        self._model = None
        self._processor = None
        self._history: list[dict[str, str]] = []  # 多轮上下文 (M5)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from scripts.infer import load_model  # 复用 infer 的加载逻辑

            self._model, self._processor = load_model(
                self.model_name, self.quant, self.adapter
            )

    @staticmethod
    def parse_json(text: str) -> dict[str, Any] | None:
        """从模型输出中提取 JSON (容忍 ```json 围栏/前后噪声)。"""
        text = text.strip()
        # 提取 ```json ... ``` 块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if m:
            text = m.group(1)
        else:
            m = re.search(r"(\{.*\})", text, re.S)
            if m:
                text = m.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def recognize(self, image: str | Path, prompt: str | None = None) -> dict[str, Any]:
        """识别一张图片, 返回结构化结果。

        返回: {"ok": bool, "data": dict|None, "raw": str, "note": str}
            - ok=True 且 data 含置信度 >= 阈值: 正常结果
            - ok=True 但置信度低: note 提示"不确定"
            - ok=False: 解析失败
        """
        from scripts.infer import recognize

        self._ensure_loaded()
        raw = recognize(
            self._model,
            self._processor,
            Path(image),
            prompt or IDENTIFY_PROMPT,
            max_tokens=512,
        )
        data = self.parse_json(raw)
        if data is None:
            return {"ok": False, "data": None, "raw": raw, "note": "模型输出无法解析为 JSON"}

        conf = float(data.get("置信度", 0.0))
        if conf < self.threshold:
            note = f"置信度较低 ({conf:.2f}), 建议换个角度/距离重新拍摄"
        else:
            note = ""
        return {"ok": True, "data": data, "raw": raw, "note": note}

    def chat(self, message: str) -> str:
        """多轮对话 (M5 实现): 基于上一轮识别结果回答追问。"""
        raise NotImplementedError("多轮对话将在 M5 阶段实现 (依赖对话状态与工具调用)")


# 便捷函数
def identify(image: str | Path, **kwargs: Any) -> dict[str, Any]:
    """一行调用: identify("photo.jpg") -> 结构化结果。"""
    agent = FloraAgent(**kwargs)
    return agent.recognize(image)
