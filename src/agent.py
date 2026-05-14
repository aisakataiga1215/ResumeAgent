"""
Resume Agent 核心
- 10 个 Tool：5 分析 + 1 RAG + 2 搜索 + 2 GitHub
- Episodic Memory: SQLite checkpointer 持久化
- 3 种模式：完整分析 / JD 发现 / GitHub 导入
"""
import re
import json
import os
from typing import Generator

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
from src.memory import KnowledgeBase, get_checkpointer
from src.tools_search import search_jobs_online, rank_jds_by_match
from src.tools_github import fetch_github_repos, generate_star_summary


def _get_llm():
    return ChatOpenAI(
        model=deepseek_model_name,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=deepseek_api_base,
        temperature=0.3,
    )

# ═══════════════════════════════════════════
# 原有 Tools (5) + RAG Tool (1)
# ═══════════════════════════════════════════

@tool
def extract_resume_info(resume_text: str) -> str:
    """从简历文本中提取结构化信息：技能列表、工作经历、项目经验、教育背景。返回 JSON。"""
    llm = _get_llm()
    prompt = f"""从以下简历文本中提取结构化信息，只返回 JSON：
{{
    "skills": ["技能1", ...],
    "work_experience": [{{"company": "", "role": "", "duration": "", "description": ""}}],
    "projects": [{{"name": "", "tech_stack": [], "description": ""}}],
    "education": [{{"school": "", "degree": "", "major": "", "year": ""}}],
    "certificates": [],
    "years_of_experience": "估算"
}}
简历：{resume_text[:8000]}"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def extract_jd_requirements(jd_text: str) -> str:
    """从职位描述(JD)中提取关键要求：必备技能、加分项、职责、资历。返回 JSON。"""
    llm = _get_llm()
    prompt = f"""从以下职位描述中提取关键要求，只返回 JSON：
{{
    "required_skills": [], "preferred_skills": [],
    "responsibilities": [], "qualifications": [],
    "experience_requirement": "", "keywords": []
}}
JD：{jd_text[:8000]}"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def analyze_dimension(dimension_name: str, resume_info: str, jd_info: str) -> str:
    """对单个评分维度深度打分。dimension_name 可选: skill_match, experience, tech_depth。返回 JSON。"""
    llm = _get_llm()
    prompts = {
        "skill_match": "对比技能与JD要求匹配度。关注技术栈重叠、技能数量、技能等级。1-10打分。",
        "experience": "对比工作/项目经历与JD职责相关性。关注行业、职责相似度、项目复杂度。1-10打分。",
        "tech_depth": "评估技术深度和广度 vs JD。关注年限、角色、技术广度。1-10打分。",
    }
    guidance = prompts.get(dimension_name, "1-10打分。")
    prompt = f"""{guidance}
简历信息：{resume_info[:4000]}
JD信息：{jd_info[:4000]}
返回 JSON：{{"dimension": "{dimension_name}", "score": 8.5, "analysis": "...", "suggestions": ["..."]}}"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def check_ats_keywords(resume_text: str, jd_keywords_json: str) -> str:
    """检查 JD 关键词在简历中的命中情况。返回命中/缺失关键词及覆盖率。"""
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
    return json.dumps({"matched": matched, "missing": missing, "hit_rate": f"{hit_rate}%", "total_keywords": total}, ensure_ascii=False)


@tool
def generate_final_report(all_scores_json: str) -> str:
    """汇总所有维度评分，生成最终报告 JSON。不需要返回 overall_score（代码计算）。"""
    llm = _get_llm()
    prompt = f"""汇总以下评分结果为最终报告 JSON：
维度权重：{json.dumps(scoring_weights, ensure_ascii=False)}
各维度结果：{all_scores_json}
格式：{{"scores": {{"skill_match": {{"name":"技能匹配度","score":8.5,"weight":0.30,"analysis":"...","suggestions":["..."]}},...}},"summary":"200字总结"}}
不要返回 overall_score 字段。建议必须具体可操作。"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def search_resume_knowledge(query: str) -> str:
    """从简历写作知识库中检索最佳实践。输入问题或关键词(如'STAR写法''ATS优化')，返回相关知识片段。"""
    kb = KnowledgeBase()
    kb.seed_if_empty()
    return kb.search(query)


# ═══════════════════════════════════════════
# ResumeAgent — 整合全部 Tool + Memory
# ═══════════════════════════════════════════

class ResumeAgent:
    def __init__(self):
        self.tools = [
            extract_resume_info,
            extract_jd_requirements,
            search_resume_knowledge,
            analyze_dimension,
            check_ats_keywords,
            search_jobs_online,
            rank_jds_by_match,
            fetch_github_repos,
            generate_star_summary,
            generate_final_report,
        ]

        # Episodic Memory: SQLite 持久化
        self.checkpointer = get_checkpointer()

        self.graph = create_agent(
            model=_get_llm(),
            tools=self.tools,
            system_prompt=resume_agent_system_prompt,
            checkpointer=self.checkpointer,
        )

    def analyze(self, resume_text: str, jd_text: str = "", github_user: str = "",
                session_id: str = "default") -> Generator[dict, None, None]:
        """执行分析，yield 中间步骤，最后 yield done。

        Mode 1: resume + jd → 完整分析
        Mode 2: resume only → 搜索 JD + 排序 + 分析
        Mode 3: resume + github_user → 导入项目 + 分析
        """
        # 构建用户消息
        parts = ["请分析以下简历"]

        if github_user:
            parts.append(f"\n用户 GitHub: {github_user}")
            parts.append("请先用 fetch_github_repos 获取项目仓库，再用 generate_star_summary 生成 STAR 描述，")
            parts.append("将 STAR 项目经历融入简历分析。")

        parts.append(f"\n=== 简历 ===\n{resume_text[:10000]}")

        if jd_text.strip():
            parts.append(f"\n=== JD ===\n{jd_text[:8000]}")
        else:
            parts.append("\n没有提供 JD，请用 search_jobs_online 搜索匹配岗位，再用 rank_jds_by_match 排序展示。")

        user_message = "\n".join(parts)
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [{"role": "user", "content": user_message}]}

        prev_msg_count = 1
        final_state = None

        for chunk in self.graph.stream(inputs, config, stream_mode="values"):
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

        output = final_state["messages"][-1].content if final_state else ""
        report = self._parse_report(output)
        report = self._recalc_overall_score(report)
        yield {"type": "done", "report": report}

    def _parse_report(self, output: str) -> dict:
        try:
            m = re.search(r"\{[\s\S]*\}", output)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"scores": {}, "overall_score": 0, "summary": output, "raw": True}

    def _recalc_overall_score(self, report: dict) -> dict:
        scores = report.get("scores", {})
        if not scores:
            return report
        ws = 0
        for key, dim in scores.items():
            if isinstance(dim, dict) and "score" in dim:
                ws += float(dim["score"]) * float(dim.get("weight", scoring_weights.get(key, 0)))
        report["overall_score"] = round(ws * 10, 1)
        if not report.get("summary"):
            report["summary"] = ""
        return report


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════
if __name__ == "__main__":
    test_resume = """
    张三 | 高级后端工程师 | 5年经验
    技能：Python, Java, Go, MySQL, Redis, Docker, K8s, Elasticsearch, Kafka
    工作经历：
    - 2020-2025 ABC科技 | 高级后端工程师 | 负责电商核心交易系统设计与开发
    - 设计并实现日均千万级订单处理系统，QPS从500提升到5000
    教育：清华大学 计算机科学与技术 硕士 2020
    """

    test_jd = """
    岗位：资深后端开发工程师
    要求：精通Java或Go，5年以上经验，熟悉MySQL、Redis、消息队列，有分布式系统设计经验
    """

    agent = ResumeAgent()
    print("开始分析 (Mode 1: resume + JD)...\n")

    for step in agent.analyze(test_resume, jd_text=test_jd, session_id="test_001"):
        if step["type"] == "done":
            r = step["report"]
            print(f"\nOverall: {r.get('overall_score')}/100")
            for k, d in r.get("scores", {}).items():
                print(f"  {d.get('name', k)}: {d.get('score')}/10")
        elif "tool" in step:
            print(f"  [{step['type']}] {step['tool']}")
