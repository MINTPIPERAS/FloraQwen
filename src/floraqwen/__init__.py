"""FloraQwen 花草识别智能体核心包。

当前包含:
- `agent.py`   : 智能体编排 (识别 + 多轮对话, 骨架)
- `schema.py`  : 结构化识别输出 (Pydantic, 可选)
"""
from .agent import FloraAgent

__version__ = "0.1.0"
__all__ = ["FloraAgent"]
