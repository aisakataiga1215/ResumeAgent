import re
import json
import os
from typing import Callable

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import config_data as config


def _get_llm():
    """创建 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model=config.deepseek_model_name,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=config.deepseek_api_base,
        temperature=0.3,
    )


# ═══════════════════════════════════════════
# Tool 定义
# ═══════════════════════════════════════════

@tool
def extract_resume_info(resume_text: str) -> str:
    """从简历文本中提取结构化信息：技能列表、工作经历、项目经验、教育背景。
    输入完整简历文本，返回 JSON 格式的结构化数据。"""
    llm = _get_llm()
    prompt = f"""从以下简历文本中提取结构化信息，只返回 JSON，不要其他内容：
{{
    "skills": ["技能1", "技能2", ...],
    "work_experience": [{{"company": "", "role": "", "duration": "", "description": ""}}, ...],
    "projects": [{{"name": "", "tech_stack": [], "description": ""}}, ...],
    "education": [{{"school": "", "degree": "", "major": "", "year": ""}}, ...],
    "certificates": ["证书1", ...],
    "years_of_experience": "总工作年限估算"
}}

简历文本：
{resume_text[:8000]}"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def extract_jd_requirements(jd_text: str) -> str:
    """从职位描述(JD)中提取关键要求：必备技能、加分项、职责、资历要求。
    输入完整 JD 文本，返回 JSON 格式的结构化要求列表。"""
    llm = _get_llm()
    prompt = f"""从以下职位描述中提取关键要求，只返回 JSON，不要其他内容：
{{
    "required_skills": ["必备技能1", "必备技能2", ...],
    "preferred_skills": ["加分技能1", ...],
    "responsibilities": ["职责1", "职责2", ...],
    "qualifications": ["学历/证书要求"],
    "experience_requirement": "经验年限要求",
    "keywords": ["ATS 高频关键词1", "关键词2", ...]
}}

JD 文本：
{jd_text[:8000]}"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def analyze_dimension(dimension_name: str, resume_info: str, jd_info: str) -> str:
    """对单个评分维度进行深度分析并打分。
    dimension_name 可选: skill_match, experience, tech_depth
    返回 JSON: {dimension, score, analysis, suggestions}"""
    llm = _get_llm()

    dimension_prompts = {
        "skill_match": "对比简历技能和 JD 技能要求，评估匹配度。关注：技术栈重叠、技能数量、技能等级。按 1-10 打分。",
        "experience": "对比简历工作/项目经历和 JD 职责要求，评估经验相关性。关注：行业匹配、职责相似度、项目复杂度。按 1-10 打分。",
        "tech_depth": "评估简历中技术掌握的深度和广度 vs JD 的要求。关注：技术使用年限、项目中的技术角色、技术广度。按 1-10 打分。",
    }
    guidance = dimension_prompts.get(dimension_name, "按 1-10 打分并给出理由和建议。")

    prompt = f"""{guidance}

简历信息：{resume_info[:4000]}

JD 信息：{jd_info[:4000]}

只返回 JSON，不要其他内容：
{{"dimension": "{dimension_name}", "score": 8.5, "analysis": "详细分析，150字左右", "suggestions": ["具体建议1", "具体建议2", "具体建议3"]}}"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def check_ats_keywords(resume_text: str, jd_keywords_json: str) -> str:
    """检查 JD 关键词在简历中的命中情况。输入简历原文和 JD 关键词 JSON，
    返回命中/缺失关键词及覆盖率。"""
    try:
        keywords_data = json.loads(jd_keywords_json)
        all_keywords = keywords_data.get("keywords", [])
        if not all_keywords:
            all_keywords = keywords_data.get("required_skills", []) + keywords_data.get("preferred_skills", [])
    except (json.JSONDecodeError, KeyError):
        all_keywords = []

    resume_lower = resume_text.lower()
    matched, missing = [], []
    for kw in all_keywords:
        if kw.lower() in resume_lower:
            matched.append(kw)
        else:
            missing.append(kw)

    total = len(all_keywords)
    hit_rate = round(len(matched) / total * 100, 1) if total > 0 else 0
    return json.dumps({
        "matched": matched,
        "missing": missing,
        "hit_rate": f"{hit_rate}%",
        "total_keywords": total
    }, ensure_ascii=False)


@tool
def generate_final_report(all_scores_json: str) -> str:
    """汇总所有维度评分和分析结果，生成最终综合报告。
    all_scores_json 是包含各维度结果的 JSON 字符串。
    返回完整的结构化分析报告 JSON。"""
    llm = _get_llm()
    weights = config.scoring_weights
    prompt = f"""以下是各维度的评分结果，请生成最终分析报告。

维度权重：{json.dumps(weights, ensure_ascii=False)}

各维度结果：
{all_scores_json}

请按以下 JSON 格式返回完整报告（只返回 JSON，不要其他内容）：
{{
    "scores": {{
        "skill_match": {{"name": "技能匹配度", "score": 8.5, "weight": 0.30, "analysis": "...", "suggestions": ["..."]}},
        "experience": {{"name": "经验相关性", "score": 7.0, "weight": 0.25, "analysis": "...", "suggestions": ["..."]}},
        "tech_depth": {{"name": "技术深度", "score": 8.0, "weight": 0.20, "analysis": "...", "suggestions": ["..."]}},
        "ats_keywords": {{"name": "关键词/ATS", "score": 6.0, "weight": 0.15, "analysis": "...", "matched": ["..."], "missing": ["..."]}},
        "overall_impression": {{"name": "综合印象", "score": 9.0, "weight": 0.10, "analysis": "评估整体结构、表达、亮点", "suggestions": ["..."]}}
    }},
    "overall_score": 78.0,
    "summary": "200字以内的综合分析总结"
}}

注意：
- 请根据已有维度结果和权重计算 overall_score（加权平均 × 10）
- overall_impression 维度需要你从简历整体结构、语言表达、亮点突出等方面评估
- 所有 suggestions 必须具体、可操作，针对简历内容提出实际修改建议"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═══════════════════════════════════════════
# ResumeAgent 主类
# ═══════════════════════════════════════════

class ResumeAgent:
    def __init__(self, on_step: Callable | None = None):
        self.llm = _get_llm()
        self.tools = [
            extract_resume_info,
            extract_jd_requirements,
            analyze_dimension,
            check_ats_keywords,
            generate_final_report,
        ]
        self.on_step = on_step

        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=config.resume_agent_system_prompt,
        )

    def analyze(self, resume_text: str, jd_text: str) -> dict:
        """执行分析，通过 stream 捕获中间步骤，返回完整报告 dict"""
        user_message = (
            f"请分析以下简历和职位描述的匹配度：\n\n"
            f"=== 简历 ===\n{resume_text[:10000]}\n\n"
            f"=== JD ===\n{jd_text[:8000]}"
        )
        inputs = {"messages": [{"role": "user", "content": user_message}]}

        # 用 stream 捕获中间步骤
        for chunk in self.graph.stream(inputs, stream_mode="updates"):
            if self.on_step:
                for node_name, node_output in chunk.items():
                    self.on_step({"node": node_name, "output": node_output})

        # 获取最终结果
        final_state = self.graph.invoke(inputs)
        messages = final_state.get("messages", [])
        output = messages[-1].content if messages else ""

        # 从输出中提取 JSON
        try:
            json_match = re.search(r"\{[\s\S]*\}", output)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {
            "scores": {},
            "overall_score": 0,
            "summary": output,
            "raw": True
        }


if __name__ == "__main__":
    test_resume = """
    张三 | 高级后端工程师 | 5年经验
    技能：Python, Java, Go, MySQL, Redis, Docker, K8s, Elasticsearch, Kafka, gRPC
    工作经历：
    - 2020-2025 ABC科技 | 高级后端工程师 | 负责电商核心交易系统设计与开发
    - 设计并实现日均千万级订单处理系统，QPS从500提升到5000
    - 主导微服务架构改造，将单体应用拆分为12个微服务
    教育：清华大学 计算机科学与技术 硕士 2020
    """

    test_jd = """
    岗位：资深后端开发工程师
    职责：
    1. 负责公司核心业务系统的架构设计与开发
    2. 参与微服务架构演进和技术选型
    3. 优化系统性能，保障高并发场景下的稳定性
    要求：
    1. 精通 Java 或 Go，5年以上开发经验
    2. 熟悉 MySQL、Redis、消息队列等中间件
    3. 有分布式系统设计经验
    4. 有电商/支付系统经验优先
    5. 熟悉 Docker、Kubernetes
    """

    print("DEEPSEEK_API_KEY =", os.environ.get("DEEPSEEK_API_KEY", "未设置")[:8] + "..." if os.environ.get("DEEPSEEK_API_KEY") else "未设置")

    def on_step(step):
        node = step.get("node", "?")
        print(f"[Agent Node] {node}")

    agent = ResumeAgent(on_step=on_step)
    print("开始分析...")
    report = agent.analyze(test_resume, test_jd)
    print("\n=== 分析报告 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
