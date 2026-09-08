# 🎓 校园新生 FAQ 助手

一个基于 **RAG（检索增强生成）** 的校园新生答疑助手：把招生简章、选课手册、宿舍规定等资料喂进去，就能用自然语言提问，AI 基于真实资料作答，并标注引用来源（可溯源、不瞎编）。

> 定位：面向 **AI 大模型应用开发岗** 的实战项目，重点展示「混合检索 + 引用溯源 + 生产化工程」而非堆砌算法名词。

---

## ✨ 功能特性

- **混合检索**：BM25 关键词匹配 + 向量语义检索，用 RRF（Reciprocal Rank Fusion）融合排序，兼顾精确命中和语义泛化。
- **引用溯源**：每个回答基于检索到的资料片段，提示词强制 LLM 用 `[1][2]` 标注来源，抑制幻觉。
- **多格式文档**：支持 PDF / Word / Markdown / 纯文本导入。
- **增量索引**：上传文档即时入库，无需重启服务。
- **大模型可插拔**：兼容 OpenAI 协议的模型（DeepSeek / 通义千问 / OpenAI / Moonshot 等）一键切换。
- **前端零依赖**：原生 HTML/CSS/JS 单页，无构建工具、无 node_modules。
- **生产化**：Docker 部署、健康检查、索引落盘、异常兜底（LLM 不可用时返回最相关原文）。

---

## 🏗️ 架构

```
┌──────────────────────────────────────────────┐
│  前端（原生 HTML/CSS/JS）                      │
│  提问 → 展示回答 + 来源                       │
└──────────────┬───────────────────────────────┘
               │ POST /api/chat
┌──────────────▼───────────────────────────────┐
│  FastAPI 后端                                 │
│  ┌─────────────────────────────────────────┐ │
│  │ Pipeline（编排）                         │ │
│  │   检索 HybridRetriever                   │ │
│  │     ├─ BM25（rank_bm25 + jieba 分词）    │ │
│  │     ├─ 向量（sentence-transformers/BGE） │ │
│  │     └─ RRF 融合                          │ │
│  │   生成 Generator（OpenAI 协议 LLM）       │ │
│  └─────────────────────────────────────────┘ │
└──────────────┬───────────────────────────────┘
               │
        ┌──────▼──────┐
        │ 文档加载/切块 │  loader + chunker
        │ 索引持久化    │  IndexStore(numpy + json)
        └─────────────┘
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY（DeepSeek 等 OpenAI 兼容 Key）
```

> 不填 Key 也能跑：问答时若大模型不可用，会自动降级为返回最相关原文。

### 3. 构建索引

```bash
python scripts/build_index.py
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload
```

打开 http://127.0.0.1:8000 即可看到聊天界面。

### Docker 一键部署

```bash
docker build -t campus-faq .
docker run -p 8000:8000 --env-file .env campus-faq
```

---

## 🔌 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查，返回已索引块数 |
| POST | `/api/chat` | 问答，入参 `{"query": "..."}` |
| POST | `/api/documents` | 上传文档（multipart 表单字段 `files`） |

`/api/chat` 响应示例：

```json
{
  "answer": "宿舍周日到周四 23:00 熄灯……[1]",
  "sources": [
    {"document": "sample_faq.md", "chunk_id": 3, "text": "……", "score": 0.0123}
  ],
  "model": "deepseek-chat"
}
```

---

## 📂 项目结构

```
campus-faq-assistant/
├── app/
│   ├── main.py               # FastAPI 入口 + 装配工厂
│   ├── config.py             # 环境变量配置
│   ├── schemas.py            # Pydantic 模型
│   ├── core/
│   │   ├── loader.py         # 文档加载（PDF/Word/文本）
│   │   ├── chunker.py        # 文本切块（段落聚合 + 重叠）
│   │   ├── embedder.py       # 向量化（惰性加载模型）
│   │   ├── retriever.py      # BM25 + 向量 + RRF 混合检索
│   │   ├── generator.py      # LLM 生成（含引用提示词）
│   │   └── pipeline.py       # 检索→生成 编排 + 兜底
│   ├── routes/chat.py        # HTTP 接口
│   └── storage/index_store.py# 索引落盘/恢复
├── scripts/
│   ├── build_index.py        # 离线构建索引
│   └── eval_rag.py           # Hit@k / MRR@k 评估
├── data/
│   ├── docs/sample_faq.md    # 示例文档（湖南中医药大学新生FAQ）
│   └── eval_questions.json   # 评估测试集
├── frontend/                 # 原生 JS 前端
├── tests/                    # 单元测试 + 接口测试
├── Dockerfile
└── requirements.txt
```

---

## 🧪 测试

单元测试不依赖大模型，使用 FakeEmbedder 注入，可离线快速运行：

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 📊 效果评估

```bash
python scripts/eval_rag.py --top-k 5
# 输出：Hit@5 与 MRR@5
```

检索评估指标（Hit@k / MRR@k）用于衡量「检索到的资料是否命中正确答案」，是 RAG 系统上线前的基础评估。

---

## 🛠️ 技术栈

Python · FastAPI · sentence-transformers（BGE 中文模型）· rank-bm25 · jieba · OpenAI SDK · numpy · PyPDF · python-docx · 原生 HTML/CSS/JS · Docker

---

## 📌 Roadmap

- [ ] 混合检索融合策略 A/B（线性加权 vs RRF）
- [ ] 重排序（rerank）提升头部命中率
- [ ] 对话历史记忆与多轮追问
- [ ] 检索质量的可视化评估报告
- [ ] 流式输出（SSE）

---

## 📄 License

[MIT](./LICENSE) © 2026 denbuild-abus
