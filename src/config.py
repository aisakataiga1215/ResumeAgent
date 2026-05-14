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
agent_max_iterations = 10

# ── 评分维度权重（总和为 1.0）──
scoring_weights = {
    "skill_match": 0.30,
    "experience": 0.25,
    "tech_depth": 0.20,
    "ats_keywords": 0.15,
    "overall_impression": 0.10,
}

# ── System Prompt ──
resume_agent_system_prompt = """你是一位资深 HR 和职业顾问，擅长分析简历与职位的匹配度。

你的分析流程：
1. 先用 extract_resume_info 提取简历中的技能、经验、项目、教育信息
2. 再用 extract_jd_requirements 提取 JD 中的关键要求
3. 然后用 analyze_dimension 逐个维度对比打分（技能匹配度、经验相关性、技术深度）
4. 用 check_ats_keywords 检查关键词覆盖情况
5. 最后用 generate_final_report 生成完整分析报告

每个维度都给出 1-10 分的评分和具体的修改建议。建议必须具体、可操作。"""


def get_api_key() -> str:
    """获取 DeepSeek API Key"""
    return os.environ.get("DEEPSEEK_API_KEY", "")
