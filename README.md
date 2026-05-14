# Resume Agent — RAG + Memory + Multi-Tool AI Agent

基于 LangGraph 的简历分析 Agent，集成 **RAG 检索、三层 Memory、10 个 Tool、实时 JD 搜索、GitHub 项目导入**。

## 架构

```
┌──────────────────────────────────────────────────┐
│                  Resume Agent                     │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Semantic │  │ Episodic │  │   Procedural     │ │
│  │ Memory   │  │ Memory   │  │   Memory         │ │
│  │ (Chroma) │  │ (SQLite) │  │ (System Prompt)  │ │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │              │                 │           │
│  ┌────┴──────────────┴─────────────────┴─────────┐ │
│  │              10 Tools                          │ │
│  │  extract_resume_info  extract_jd_requirements  │ │
│  │  search_resume_knowledge (RAG)                 │ │
│  │  analyze_dimension    check_ats_keywords       │ │
│  │  search_jobs_online   rank_jds_by_match        │ │
│  │  fetch_github_repos   generate_star_summary    │ │
│  │  generate_final_report                         │ │
│  └────────────────────────────────────────────────┘ │
│                       │                             │
│                  LLM (DeepSeek)                      │
└──────────────────────────────────────────────────────┘
```

## Memory 三层设计

| 类型 | 存储 | 实现 | 作用 |
|------|------|------|------|
| **Semantic** (知识) | Chroma 向量库 | `src/memory.py` | 简历写作最佳实践、STAR 模板、行业关键词 → Tool 检索 |
| **Episodic** (经历) | SQLite | `src/memory.py` → checkpointer | 记住用户历史分析、JD 偏好 |
| **Procedural** (流程) | System Prompt | `src/config.py` | 分析步骤定义、多模式切换指令 |

## 三种模式

| 模式 | 输入 | 流程 |
|------|------|------|
| **完整分析** | 简历 PDF + JD | Agent 直接分析 → 5 维评分 + 建议 |
| **JD 发现** | 简历 PDF (无JD) | 搜索岗位 → 排序展示 → 选择 → 分析 |
| **GitHub 导入** | 简历 PDF + GitHub 用户名 | 获取 repos → STAR 总结 → 融入简历 → 分析 |

## 10 个 Tools

### 分析类 (`src/agent.py`)
| Tool | 功能 |
|------|------|
| `extract_resume_info` | 从简历提取结构化信息（技能/经历/项目/教育） |
| `extract_jd_requirements` | 从 JD 提取要求（技能/职责/资历/关键词） |
| `analyze_dimension` | 单维度深度打分（技能/经验/深度） |
| `check_ats_keywords` | JD 关键词命中检查 + 覆盖率 |
| `generate_final_report` | 汇总生成完整分析报告 |

### RAG 检索 (`src/agent.py` ← `src/memory.py`)
| Tool | 功能 |
|------|------|
| `search_resume_knowledge` | 从 Chroma 知识库检索简历写作最佳实践 |

### JD 搜索 (`src/tools_search.py`)
| Tool | 功能 |
|------|------|
| `search_jobs_online` | DuckDuckGo 实时搜索招聘岗位 |
| `rank_jds_by_match` | LLM 对搜索结果按简历匹配度排序 |

### GitHub 集成 (`src/tools_github.py`)
| Tool | 功能 |
|------|------|
| `fetch_github_repos` | 获取 GitHub 用户公开仓库 |
| `generate_star_summary` | 仓库 → STAR 法则项目描述 |

## 项目结构

```
ResumeAgent/
├── app.py                    # Streamlit 入口
├── src/
│   ├── config.py             # 统一配置 + .env 加载
│   ├── agent.py              # Agent 核心（10 tools + checkpointer）
│   ├── memory.py             # Chroma KB + SQLite checkpointer
│   ├── pdf_parser.py         # PDF 文本提取
│   ├── tools_search.py       # JD 搜索工具
│   ├── tools_github.py       # GitHub API 工具
│   └── rag/                  # 原 RAG 知识库项目（归档）
├── .env.example
├── requirements.txt
└── README.md
```

## 快速开始

### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 GITHUB_TOKEN（可选）
```

### 3) 启动
```bash
streamlit run app.py
```

## 技术栈

- **Agent 框架**: LangGraph (`create_agent`)
- **LLM**: DeepSeek (`deepseek-chat`)
- **Memory**: Chroma (向量知识库) + SQLite (对话历史)
- **Embedding**: `all-MiniLM-L6-v2` (本地，无需 API)
- **搜索**: DuckDuckGo (免费)
- **GitHub**: PyGithub
- **UI**: Streamlit
