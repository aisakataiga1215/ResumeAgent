"""
GitHub API 工具
- fetch_github_repos: 获取用户公开仓库
- generate_star_summary: 将仓库转为 STAR 法则项目描述
"""
import json
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from github import Github, GithubException

from src.config import deepseek_model_name, deepseek_api_base


def _get_llm():
    return ChatOpenAI(
        model=deepseek_model_name,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=deepseek_api_base,
        temperature=0.3,
    )


def _get_github_client():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return None
    return Github(token)


@tool
def fetch_github_repos(username: str) -> str:
    """获取GitHub用户的公开仓库列表。输入GitHub用户名，返回仓库的JSON列表，
    包含名称、描述、语言、星数、Fork数、Topics。"""
    g = _get_github_client()
    if g is None:
        return json.dumps(
            {"error": "GitHub Token 未配置，请设置 GITHUB_TOKEN 环境变量"},
            ensure_ascii=False
        )

    try:
        user = g.get_user(username)
        repos = user.get_repos()

        repo_list = []
        for repo in repos:
            if repo.fork:
                continue  # 跳过 fork 的仓库
            repo_list.append({
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description or "",
                "language": repo.language or "",
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "topics": repo.get_topics() or [],
                "url": repo.html_url,
            })

        repo_list.sort(key=lambda r: r["stars"], reverse=True)
        top_repos = repo_list[:10]

        return json.dumps({
            "username": username,
            "total_repos": len(repo_list),
            "repos": top_repos,
        }, ensure_ascii=False)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API 错误: {e.data.get('message', str(e))}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def generate_star_summary(repo_json: str) -> str:
    """将 GitHub 仓库信息转为 STAR 法则项目描述。
    输入仓库 JSON 列表，返回每个仓库的 STAR 描述：项目名称、技术栈、角色、动作、成果。"""
    llm = _get_llm()

    try:
        data = json.loads(repo_json)
        repos = data.get("repos", [])
    except (json.JSONDecodeError, KeyError):
        return json.dumps({"error": "仓库数据格式错误"}, ensure_ascii=False)

    if not repos:
        return json.dumps({"projects": [], "message": "无仓库数据"}, ensure_ascii=False)

    repo_summaries = []
    for r in repos:
        repo_summaries.append(
            f"项目：{r.get('name','')} | 语言：{r.get('language','')} | "
            f"描述：{r.get('description','')[:200]} | "
            f"Topics：{','.join(r.get('topics',[]))} | Star数：{r.get('stars',0)}"
        )

    prompt = f"""将以下 GitHub 开源项目转为中文简历的 STAR 法则项目描述。
每项格式：项目名(技术栈) | S:背景 T:目标 A:个人贡献 R:量化成果。
成果可参考 Star 数和项目规模。用第一人称"我"描述。只返回 JSON：

{chr(10).join(repo_summaries)}

返回格式：
{{
    "projects": [
        {{"name": "...", "tech_stack": ["..."], "star_description": "..."}},
        ...
    ]
}}"""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    print("=== fetch_github_repos ===")
    result = fetch_github_repos.invoke({"username": "aisakataiga1215"})
    print(result[:500])

    if not json.loads(result).get("error"):
        print("\n=== generate_star_summary ===")
        star = generate_star_summary.invoke({"repo_json": result})
        print(star[:800])
