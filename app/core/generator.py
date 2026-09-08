"""答案生成：调用兼容 OpenAI 协议的大模型，基于检索资料作答并标注引用。"""
from typing import List

from openai import OpenAI

SYSTEM_PROMPT = (
    "你是校园新生答疑助手。请仅依据下方提供的【资料】回答用户问题，"
    "不要使用外部知识，不要编造。回答时在引用资料处用 [1][2] 这样的编号标注来源。"
    "如果资料不足以回答，请直接说明“现有资料无法回答该问题”，并给出建议。"
    "回答用中文，简洁清晰。"
)


class Generator:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        # 未配置 Key 时不建客户端，调用时抛出可读错误，由上层兜底
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

    def generate(self, query: str, contexts: List[dict]) -> str:
        if self.client is None:
            raise RuntimeError("未配置 LLM_API_KEY，无法调用大模型")

        parts = [f"[{i}] {ctx['text']}" for i, ctx in enumerate(contexts, start=1)]
        context_block = "\n\n".join(parts)
        user_msg = f"【资料】\n{context_block}\n\n【问题】\n{query}"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
