import re
import json
import os
from typing import Generator, Any

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.config import (
    deepseek_model_name,
    deepseek_api_base,
    scoring_weights,
    agent_max_iterations,
    resume_agent_system_prompt,
)


def _get_llm():
    return ChatOpenAI(
        model=deepseek_model_name,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=deepseek_api_base,
        temperature=0.3,
    )


# ═══════════════════════════════════════════
# Tools
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
    prompts = {
        "skill_match": "对比简历技能和 JD 技能要求，评估匹配度。关注：技术栈重叠、技能数量、技能等级。按 1-10 打分。",
        "experience": "对比简历工作/项目经历和 JD 职责要求，评估经验相关性。关注：行业匹配、职责相似度、项目复杂度。按 1-10 打分。",
        "tech_depth": "评估简历中技术掌握的深度和广度 vs JD 的要求。关注：技术使用年限、项目中的技术角色、技术广度。按 1-10 打分。",
    }
    guidance = prompts.get(dimension_name, "按 1-10 打分并给出理由和建议。")
    prompt = f"""{guidance}

简历信息：{resume_info[:4000]}

JD 信息：{jd_info[:4000]}

只返回 JSON：{{"dimension": "{dimension_name}", "score": 8.5, "analysis": "详细分析", "suggestions": ["建议1", "建议2", "建议3"]}}"""
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
        kw_data = json.loads(jd_keywords_json)
        all_kw = kw_data.get("keywords", [])
        if not all_kw:
            all_kw = kw_data.get("required_skills", []) + kw_data.get("preferred_skills", [])
    except (json.JSONDecodeError, KeyError):
        all_kw = []

    resume_lower = resume_text.lower()
    matched = [kw for kw in all_kw if kw.lower() in resume_lower]
    missing = [kw for kw in all_kw if kw.lower() not in resume_lower]
    total = len(all_kw)
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
    all_scores_json 是包含各维度结果的 JSON 字符串。返回完整报告 JSON。"""
    llm = _get_llm()
    prompt = f"""以下是各维度的评分结果，请生成最终分析报告。

维度权重：{json.dumps(scoring_weights, ensure_ascii=False)}

各维度结果：
{all_scores_json}

请按以下 JSON 格式返回完整报告（只返回 JSON）：
{{
    "scores": {{
        "skill_match": {{"name": "技能匹配度", "score": 8.5, "weight": 0.30, "analysis": "...", "suggestions": ["..."]}},
        "experience": {{"name": "经验相关性", "score": 7.0, "weight": 0.25, "analysis": "...", "suggestions": ["..."]}},
        "tech_depth": {{"name": "技术深度", "score": 8.0, "weight": 0.20, "analysis": "...", "suggestions": ["..."]}},
        "ats_keywords": {{"name": "关键词/ATS", "score": 6.0, "weight": 0.15, "analysis": "...", "matched": ["..."], "missing": ["..."]}},
        "overall_impression": {{"name": "综合印象", "score": 9.0, "weight": 0.10, "analysis": "评估整体结构、表达、亮点", "suggestions": ["..."]}}
    }},
    "summary": "200字以内的综合分析总结"
}}

注意：不需要返回 overall_score 字段，评分汇总由代码计算。
所有 suggestions 必须具体、可操作，针对简历内容提出实际修改建议。"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═══════════════════════════════════════════
# ResumeAgent
# ═══════════════════════════════════════════

class ResumeAgent:
    def __init__(self):
        self.tools = [
            extract_resume_info,
            extract_jd_requirements,
            analyze_dimension,
            check_ats_keywords,
            generate_final_report,
        ]

        self.graph = create_agent(
            model=_get_llm(),
            tools=self.tools,
            system_prompt=resume_agent_system_prompt,
        )

    def analyze(self, resume_text: str, jd_text: str) -> Generator[dict, None, dict]:
        """执行分析，yield 每个中间步骤，return 最终报告。

        Fix #1: 只 stream 一次，最后一帧即为完整最终状态（不再 invoke）。
        Fix #3: overall_score 由 Python 确定性计算，不依赖 LLM。
        """
        user_message = (
            f"请分析以下简历和职位描述的匹配度：\n\n"
            f"=== 简历 ===\n{resume_text[:10000]}\n\n"
            f"=== JD ===\n{jd_text[:8000]}"
        )
        inputs = {"messages": [{"role": "user", "content": user_message}]}

        prev_msg_count = 1  # 初始消息数（user message）
        final_state = None

        for chunk in self.graph.stream(inputs, stream_mode="values"):
            final_state = chunk
            messages = chunk.get("messages", [])
            new_msgs = messages[prev_msg_count:]

            for msg in new_msgs:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {"type": "tool_start", "tool": tc.get("name", "?"), "args": str(tc.get("args", {}))[:150]}
                elif isinstance(msg, ToolMessage):
                    yield {"type": "tool_end", "tool": getattr(msg, "name", "tool"), "preview": str(msg.content)[:200]}
                elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    yield {"type": "model_msg", "content": msg.content[:200]}

            prev_msg_count = len(messages)

        # 最终状态已在最后一次 values 中
        output = final_state["messages"][-1].content if final_state else ""

        # 提取 JSON 报告
        report = self._parse_report(output)

        # Fix #3: 用 Python 确定性计算 overall_score
        report = self._recalc_overall_score(report)

        yield {"type": "done", "report": report}

    def _parse_report(self, output: str) -> dict:
        try:
            json_match = re.search(r"\{[\s\S]*\}", output)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"scores": {}, "overall_score": 0, "summary": output, "raw": True}

    def _recalc_overall_score(self, report: dict) -> dict:
        """代码计算 overall_score，不信任 LLM 的算术"""
        scores = report.get("scores", {})
        if not scores:
            return report

        weighted_sum = 0
        for key, dim_data in scores.items():
            if isinstance(dim_data, dict) and "score" in dim_data:
                score = float(dim_data["score"])
                weight = float(dim_data.get("weight", scoring_weights.get(key, 0)))
                weighted_sum += score * weight

        report["overall_score"] = round(weighted_sum * 10, 1)
        if not report.get("summary"):
            report["summary"] = ""
        return report


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════
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

    print("DEEPSEEK_API_KEY =",
          (os.environ.get("DEEPSEEK_API_KEY", "")[:8] + "...") if os.environ.get("DEEPSEEK_API_KEY") else "未设置")

    agent = ResumeAgent()
    print("开始分析...\n")

    final_report = None
    for step in agent.analyze(test_resume, test_jd):
        if step["type"] == "done":
            final_report = step["report"]
        else:
            print(f"  [{step['type']}] {step.get('tool', step.get('content', ''))[:80]}")

    print("\n=== 分析报告 ===")
    print(f"Overall Score: {final_report.get('overall_score', 'N/A')} / 100")
    for key, dim in final_report.get("scores", {}).items():
        print(f"  {dim.get('name', key)}: {dim.get('score', '?')}/10 (×{int(dim.get('weight',0)*100)}%)")
    print(f"\nSummary: {final_report.get('summary', 'N/A')[:200]}")
