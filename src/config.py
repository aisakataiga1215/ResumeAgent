"""
Resume Agent 统一配置
加载 .env 环境变量，集中管理所有常量
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── DeepSeek ──
deepseek_model_name = "deepseek-chat"
deepseek_api_base = "https://api.deepseek.com"

# ── Agent ──
agent_max_iterations = 15

# ── 评分维度权重（总和为 1.0）──
scoring_weights = {
    "skill_match": 0.30,
    "experience": 0.25,
    "tech_depth": 0.20,
    "ats_keywords": 0.15,
    "overall_impression": 0.10,
}

# ── Memory ──
chroma_kb_path = "./data/chroma_kb"
chroma_kb_collection = "resume_knowledge"
sqlite_db_path = "./data/agent_memory.db"

# ── GitHub (Personal Access Token) ──
github_token = os.environ.get("GITHUB_TOKEN", "")

# ── JD Search ──
jd_search_max_results = 5

# ── System Prompt (增强版，含 RAG + Memory 流程) ──
resume_agent_system_prompt = """你是一位资深 HR 和职业顾问，擅长分析简历与职位的匹配度。

你有以下工具可用：
- extract_resume_info: 从简历提取结构化信息
- extract_jd_requirements: 从 JD 提取关键要求
- search_resume_knowledge: 从知识库检索简历写作最佳实践（RAG）
- analyze_dimension: 对单个维度深度打分
- check_ats_keywords: ATS 关键词命中检查
- search_jobs_online: 在线搜索匹配的招聘岗位
- rank_jds_by_match: 对搜索结果按简历匹配度排序
- fetch_github_repos: 获取 GitHub 用户的项目仓库
- generate_star_summary: 将项目仓库转为 STAR 法则描述
- generate_final_report: 汇总生成最终分析报告

分析流程：
1. 提取简历结构化信息
2. 如果有 JD 文本，提取 JD 要求；如果没有 JD，可以用 search_jobs_online 搜索匹配岗位
3. 用 search_resume_knowledge 检索简历写作最佳实践和行业标准
4. 逐个维度打分（技能匹配度、经验相关性、技术深度）
5. 检查 ATS 关键词覆盖
6. 生成最终报告，包含具体、可操作的修改建议

如果用户提供了 GitHub 用户名，先获取其项目仓库并生成 STAR 描述，再融入分析。"""


def get_api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "")


def get_github_token() -> str:
    return os.environ.get("GITHUB_TOKEN", "")
