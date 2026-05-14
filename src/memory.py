"""
Memory 管理：Semantic Memory (Chroma) + Episodic Memory (SQLite)

- Semantic: Chroma 向量库存储简历写作知识，Agent 通过 search_resume_knowledge tool 检索
- Episodic: LangGraph SqliteSaver 持久化对话历史，记住用户偏好
"""
import os
import json
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import chroma_kb_path, chroma_kb_collection, sqlite_db_path

# ═══════════════════════════════════════════
# Semantic Memory — Chroma 知识库
# ═══════════════════════════════════════════

# 种子知识数据（中文简历写作指南 + STAR 模板 + 技术栈关键词）
SEED_KNOWLEDGE = [
    # ── 简历写作最佳实践 ──
    {
        "content": "简历技能部分要点：按熟练度分级（精通/熟练/了解），与JD关键词一一对应。"
                  "技术栈按类别分组（后端/前端/数据库/运维/工具），每组3-5项。"
                  "示例：后端: Python(精通), Java(熟练), Go(了解) | 数据库: MySQL, Redis, PostgreSQL",
        "category": "best_practice",
        "tags": "技能 写法 格式",
    },
    {
        "content": "工作经历使用 STAR 法则：Situation(背景) - Task(任务) - Action(行动) - Result(结果)。"
                  "每段经历控制在3-4行，用数据量化成果。"
                  "示例：负责电商核心交易系统(S)，日均千万级订单处理(T)，主导微服务架构改造将单体拆分为12个微服务(A)，"
                  "QPS从500提升至5000，系统可用性达99.99%(R)。",
        "category": "best_practice",
        "tags": "STAR 工作经历 写法",
    },
    {
        "content": "项目经验写法：项目名称 + 技术栈 + 你的角色 + 解决的问题 + 量化成果。"
                  "避免只列职责，要突出个人贡献和技术难点。"
                  "差：参与了用户系统开发。"
                  "好：设计并实现高并发用户认证系统(Go+Redis+JWT)，支持10万QPS，将登录延迟从800ms降至50ms。",
        "category": "best_practice",
        "tags": "项目经验 写法 量化",
    },
    {
        "content": "简历整体结构：个人信息 → 技术栈 → 工作经历(倒序) → 项目经验 → 教育背景 → 证书/其他。"
                  "长度控制在1-2页，最重要的信息放在前1/3。"
                  "使用简洁的无衬线字体，避免花哨模板。ATS系统更易解析纯文本格式。",
        "category": "best_practice",
        "tags": "结构 格式 ATS",
    },
    # ── STAR 模板 ──
    {
        "content": "STAR 项目描述模板：基于[技术栈]开发了[系统名称]，解决了[业务痛点]。"
                  "承担[角色]，负责[核心模块]。通过[技术方案]，实现了[量化成果]。"
                  "项目地址：https://github.com/xxx/xxx",
        "category": "template",
        "tags": "STAR 模板 项目",
    },
    {
        "content": "GitHub 开源项目 STAR 转化：(S) 该项目是一个[功能描述]工具/框架/应用，"
                  "(T) 解决了[目标用户]在[场景]下的[痛点]，"
                  "(A) 采用[语言+核心库]实现，个人负责[模块/功能]，运用了[技术方案]，"
                  "(R) 获得[N]个Star，被[M]人使用，性能提升[X]%。",
        "category": "template",
        "tags": "GitHub STAR 开源",
    },
    # ── 行业关键词 ──
    {
        "content": "后端开发 JD 高频关键词：Java, Spring Boot, MyBatis, Go, Gin, Python, Django, FastAPI, "
                  "MySQL, PostgreSQL, Redis, MongoDB, Elasticsearch, Kafka, RabbitMQ, gRPC, RESTful API, "
                  "微服务, 分布式系统, Docker, Kubernetes, CI/CD, 高并发, 性能优化, 系统架构设计。",
        "category": "keywords",
        "tags": "后端 JD 关键词",
    },
    {
        "content": "AI/ML 岗位 JD 高频关键词：Python, PyTorch, TensorFlow, Transformer, LLM, RAG, "
                  "LangChain, Agent, NLP, CV, 深度学习, 模型训练, 模型部署, MLOps, CUDA, ONNX, "
                  "Prompt Engineering, Fine-tuning, Vector Database, Embedding。",
        "category": "keywords",
        "tags": "AI ML JD 关键词",
    },
    {
        "content": "前端开发 JD 高频关键词：JavaScript, TypeScript, React, Vue, Angular, Next.js, "
                  "Node.js, Webpack, Vite, CSS3, Tailwind, Redux, Zustand, HTTP/HTTPS, "
                  "性能优化, 跨浏览器兼容, 响应式设计, WebAssembly。",
        "category": "keywords",
        "tags": "前端 JD 关键词",
    },
    {
        "content": "ATS 简历优化要点：1) 使用标准章节标题(Work Experience而非My Journey)；"
                  "2) 关键词要精确匹配JD用语(JD写'Kubernetes'不要写'K8s')；"
                  "3) 避免表格、图片、特殊符号；4) 以纯文本格式保存一份副本用于ATS投递；"
                  "5) 技能部分单独列出，方便ATS解析。",
        "category": "best_practice",
        "tags": "ATS 优化 关键词匹配",
    },
]


class KnowledgeBase:
    """Semantic Memory: Chroma 向量知识库"""

    def __init__(self):
        os.makedirs(chroma_kb_path, exist_ok=True)
        self.embedding = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.store = Chroma(
            collection_name=chroma_kb_collection,
            embedding_function=self.embedding,
            persist_directory=chroma_kb_path,
        )

    def seed_if_empty(self):
        """如果知识库为空，写入种子数据"""
        existing = self.store.get()
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            return  # 已有数据

        texts = [item["content"] for item in SEED_KNOWLEDGE]
        metadatas = [{"category": item["category"], "tags": item["tags"]} for item in SEED_KNOWLEDGE]
        self.store.add_texts(texts, metadatas=metadatas)
        print(f"[KnowledgeBase] 已写入 {len(texts)} 条种子知识")

    def search(self, query: str, k: int = 3) -> str:
        """检索相关知识，返回 JSON 字符串"""
        docs = self.store.similarity_search(query, k=k)
        if not docs:
            return json.dumps({"results": [], "query": query}, ensure_ascii=False)

        results = []
        for d in docs:
            results.append({
                "content": d.page_content[:300],
                "category": d.metadata.get("category", ""),
                "tags": d.metadata.get("tags", ""),
            })
        return json.dumps({"results": results, "query": query}, ensure_ascii=False)


# ═══════════════════════════════════════════
# Episodic Memory — SQLite Checkpointer
# ═══════════════════════════════════════════

def get_checkpointer() -> SqliteSaver:
    """创建 SQLite 持久化 checkpointer"""
    os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
    return SqliteSaver.from_conn_string(sqlite_db_path)


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("=== Semantic Memory Test ===")
    kb = KnowledgeBase()
    kb.seed_if_empty()
    result = kb.search("STAR法则怎么写项目经历")
    print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))

    print("\n=== Episodic Memory Test ===")
    checkpointer = get_checkpointer()
    print(f"SQLite checkpointer created at: {sqlite_db_path}")
    print("OK")
