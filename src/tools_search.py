"""
JD 实时搜索工具
- search_jobs_online: DuckDuckGo 搜索招聘岗位
- rank_jds_by_match: 对搜索结果按简历匹配度排序
- fetch_jd_from_url: 从URL抓取完整JD内容
"""
import json
import re
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from ddgs import DDGS

from src.config import deepseek_model_name, deepseek_api_base, jd_search_max_results
import os


def _get_llm():
    return ChatOpenAI(
        model=deepseek_model_name,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=deepseek_api_base,
        temperature=0.3,
    )


@tool
def search_jobs_online(query: str) -> str:
    """在网络上搜索招聘岗位(JD)。输入搜索关键词(如"Python后端工程师 北京")，
    返回 JSON 格式的岗位列表，包含标题、公司、摘要、URL。"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"{query} 招聘 岗位要求",
                max_results=jd_search_max_results
            ))
            jobs = []
            for r in results:
                jobs.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", "")[:300],
                    "url": r.get("href", ""),
                })
            return json.dumps({"query": query, "jobs": jobs, "count": len(jobs)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)


@tool
def rank_jds_by_match(resume_info: str, jd_list_json: str) -> str:
    """根据简历信息对 JD 列表进行匹配度排序。输入简历结构化信息和 JD 列表 JSON，
    返回按匹配度从高到低排序的 JD 列表，每个 JD 附带匹配分(0-100)和匹配理由。"""
    llm = _get_llm()

    try:
        jd_data = json.loads(jd_list_json)
        jd_items = jd_data.get("jobs", [])[:10]
    except (json.JSONDecodeError, KeyError):
        return json.dumps({"error": "JD 列表格式错误"}, ensure_ascii=False)

    if not jd_items:
        return json.dumps({"results": [], "message": "无 JD 可供排序"}, ensure_ascii=False)

    jd_summaries = []
    for i, jd in enumerate(jd_items):
        jd_summaries.append(f"[{i}] {jd.get('title','')} | {jd.get('snippet','')[:200]}")

    prompt = f"""根据求职者简历信息，对以下 JD 进行匹配度打分排序。只返回 JSON：

简历信息：
{resume_info[:3000]}

JD 列表：
{chr(10).join(jd_summaries)}

返回格式（只返回 JSON）：
{{
    "ranked": [
        {{"index": 0, "title": "...", "match_score": 95, "reason": "一句话匹配理由"}},
        ...
    ]
}}"""
    try:
        resp = llm.invoke(prompt)
        ranked = json.loads(resp.content)
        # LLM 可能直接返回列表，统一转为 {"ranked": [...]}
        if isinstance(ranked, list):
            ranked = {"ranked": ranked}
        # 把原始 JD 信息合并回去
        for item in ranked.get("ranked", []):
            idx = item.get("index", 0)
            if idx < len(jd_items):
                item["snippet"] = jd_items[idx].get("snippet", "")[:300]
                item["url"] = jd_items[idx].get("url", "")
        return json.dumps(ranked, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def fetch_jd_from_url(url: str) -> str:
    """从 URL 抓取页面内容，提取 JD 文本。使用简单的 HTML 清洗。"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = resp.apparent_encoding or "utf-8"

        # 简单 HTML 清洗：移除 script/style/tags
        html = resp.text
        for tag in ["script", "style", "nav", "footer", "header"]:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", "\n", html)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # 截取合理长度（前8000字符）
        return text[:8000].strip()
    except Exception as e:
        return f"[获取失败] {e}"


if __name__ == "__main__":
    test_resume_info = """
    技能：Python, Java, Go, MySQL, Redis, Docker, K8s
    经验：5年后端开发，电商系统架构
    """

    print("=== search_jobs_online ===")
    result = search_jobs_online.invoke({"query": "Python 后端工程师 远程"})
    print(result[:500])

    print("\n=== rank_jds_by_match ===")
    ranked = rank_jds_by_match.invoke({
        "resume_info": test_resume_info,
        "jd_list_json": result
    })
    print(ranked[:500])
