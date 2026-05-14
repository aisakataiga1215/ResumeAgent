import os, json
import streamlit as st
from src.pdf_parser import extract_text_from_pdf
from src.agent import ResumeAgent
from src.tools_search import search_jobs_online, rank_jds_by_match
from src.tools_github import fetch_github_repos, generate_star_summary
from src import config


st.set_page_config(
    page_title="Resume Agent — RAG + Memory",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg, #0f1117 0%, #1a1d28 50%, #151820 100%); }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 3.2rem; color: #F0C060; letter-spacing: -0.02em; margin-bottom: 0; }
.hero-sub { font-size: 1rem; color: #6b7280; letter-spacing: 0.08em; text-transform: uppercase; }
.overall-card { background: linear-gradient(135deg, rgba(240,192,96,0.12), rgba(240,192,96,0.04)); border: 1px solid rgba(240,192,96,0.25); border-radius: 20px; padding: 32px 40px; text-align: center; backdrop-filter: blur(12px); }
.overall-score { font-family: 'DM Serif Display', serif; font-size: 5rem; color: #F0C060; line-height: 1; margin: 0; }
.overall-label { color: #9ca3af; font-size: 0.9rem; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 8px; }
.dim-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px 24px; backdrop-filter: blur(8px); transition: transform 0.15s; }
.dim-card:hover { transform: translateY(-2px); border-color: rgba(240,192,96,0.35); }
.dim-name { font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
.dim-score { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #f1f5f9; }
.dim-weight { font-size: 0.75rem; color: #6b7280; }
.dim-bar-bg { background: rgba(255,255,255,0.06); border-radius: 6px; height: 6px; margin-top: 10px; overflow: hidden; }
.dim-bar-fill { border-radius: 6px; height: 100%; transition: width 0.6s ease; }
.jd-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 18px 20px; margin-bottom: 10px; transition: 0.15s; }
.jd-card:hover { border-color: rgba(240,192,96,0.4); }
.suggestion-high { border-left: 3px solid #ef4444; }
.suggestion-mid  { border-left: 3px solid #f59e0b; }
.suggestion-low  { border-left: 3px solid #10b981; }
.suggestion-item { background: rgba(255,255,255,0.02); border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; font-size: 0.92rem; color: #d1d5db; }
[data-testid="stSidebar"] { background: rgba(15,17,23,0.8); border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] h3 { color: #F0C060 !important; font-size: 0.85rem; letter-spacing: 0.06em; text-transform: uppercase; }
div.stButton > button { background: linear-gradient(135deg, #F0C060, #d4952a) !important; color: #0f1117 !important; border: none !important; border-radius: 10px !important; padding: 12px 32px !important; font-weight: 600 !important; font-size: 1rem !important; letter-spacing: 0.04em !important; text-transform: uppercase !important; width: 100% !important; }
div.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 24px rgba(240,192,96,0.25) !important; }
[data-testid="stFileUploader"] { border: 1px dashed rgba(255,255,255,0.15) !important; border-radius: 12px !important; background: rgba(255,255,255,0.02) !important; }
hr { border-color: rgba(255,255,255,0.06) !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════
for k, v in {
    "report": None, "thinking_log": [], "analyzing": False,
    "resume_text": "", "jd_text": "",
    "searched_jds": None, "selected_jd": None,
    "github_projects": None, "github_repo_data": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_score_color(score):
    if score >= 8: return "#10b981"
    elif score >= 6: return "#f59e0b"
    return "#ef4444"


def render_score_card(dim_data, col):
    score = dim_data.get("score", 0)
    name = dim_data.get("name", "?")
    weight = dim_data.get("weight", 0)
    color = get_score_color(score)
    with col:
        st.markdown(f"""
        <div class="dim-card">
            <div class="dim-name">{name} <span class="dim-weight">x {int(weight*100)}%</span></div>
            <div class="dim-score" style="color:{color}">{score:.1f}<span style="font-size:1rem;color:#6b7280">/10</span></div>
            <div class="dim-bar-bg"><div class="dim-bar-fill" style="width:{int(score*10)}%;background:linear-gradient(90deg,{color},{color}cc)"></div></div>
            <p style="color:#9ca3af;font-size:0.82rem;margin-top:10px;">{dim_data.get('analysis','')[:120]}</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📤 上传简历")
    uploaded_file = st.file_uploader("支持 PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        try:
            text = extract_text_from_pdf(uploaded_file.getvalue())
            st.session_state.resume_text = text
            with st.expander(f"📄 预览 ({len(text)} 字符)", expanded=False):
                st.caption(text[:500] + ("..." if len(text) > 500 else ""))
            if len(text) < 50:
                st.warning("内容过短，可能是扫描件")
        except Exception as e:
            st.error(f"解析失败: {e}")

    st.markdown("---")

    # ── JD 输入（可选）──
    st.markdown("### 📋 职位描述 (可选)")
    jd_input = st.text_area("粘贴 JD", height=150, placeholder="粘贴 JD，或留空使用搜索功能...", label_visibility="collapsed")
    st.session_state.jd_text = jd_input.strip()

    # ── JD 搜索（无 JD 时）──
    if not st.session_state.jd_text:
        st.markdown("### 🔍 或搜索匹配岗位")
        search_query = st.text_input("搜索关键词", placeholder="如: Python后端 远程")
        if st.button("🔍 搜索岗位", use_container_width=True) and search_query:
            if not st.session_state.resume_text:
                st.error("请先上传简历")
            else:
                with st.spinner("搜索中..."):
                    raw = search_jobs_online.invoke({"query": search_query})
                    st.session_state.searched_jds = json.loads(raw)
        st.markdown("---")

    # ── GitHub 导入（可选）──
    st.markdown("### 🐙 GitHub 导入 (可选)")
    gh_user = st.text_input("GitHub 用户名", placeholder="如: aisakataiga1215", label_visibility="collapsed")
    if st.button("🐙 获取项目", use_container_width=True) and gh_user:
        if not config.get_github_token():
            st.error("请设置 GITHUB_TOKEN")
        else:
            with st.spinner("获取 GitHub 仓库..."):
                raw = fetch_github_repos.invoke({"username": gh_user})
                data = json.loads(raw)
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.session_state.github_repo_data = data
                    with st.spinner("生成 STAR 描述..."):
                        star_raw = generate_star_summary.invoke({"repo_json": raw})
                        st.session_state.github_projects = json.loads(star_raw)
                    st.success(f"获取 {len(data.get('repos',[]))} 个仓库")
    st.markdown("---")

    # ── 权重 ──
    with st.expander("⚙ 权重调整"):
        w = {}
        w["skill_match"] = st.slider("技能匹配", 10, 50, int(config.scoring_weights["skill_match"]*100), 5, format="%d%%")
        w["experience"] = st.slider("经验相关", 10, 50, int(config.scoring_weights["experience"]*100), 5, format="%d%%")
        w["tech_depth"] = st.slider("技术深度", 10, 50, int(config.scoring_weights["tech_depth"]*100), 5, format="%d%%")
        w["ats_keywords"] = st.slider("ATS关键词", 5, 30, int(config.scoring_weights["ats_keywords"]*100), 5, format="%d%%")
        w["overall_impression"] = st.slider("综合印象", 5, 30, int(config.scoring_weights["overall_impression"]*100), 5, format="%d%%")
        if sum(w.values()) != 100:
            st.warning(f"合计 {sum(w.values())}%，应为100%")
        config.scoring_weights = {k: v/100 for k, v in w.items()}

    st.markdown("---")
    analyze_btn = st.button("🚀 开始分析", disabled=st.session_state.analyzing)
    st.caption("Powered by DeepSeek · RAG + Memory")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
st.markdown('<p class="hero-title">Resume Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">RAG + Memory · 10 Tools · 3 Modes</p>', unsafe_allow_html=True)
st.divider()

# ── JD 搜索结果 ──
if st.session_state.searched_jds and not st.session_state.report:
    st.markdown("### 🔍 匹配岗位列表")
    jobs = st.session_state.searched_jds.get("jobs", [])
    if jobs:
        for i, jd in enumerate(jobs[:5]):
            snippet = jd.get("snippet", "")[:150]
            title = jd.get("title", "未知岗位")
            with st.container():
                st.markdown(f"""
                <div class="jd-card">
                    <strong style="color:#F0C060">{title}</strong><br>
                    <span style="color:#9ca3af;font-size:0.85rem;">{snippet}</span>
                </div>""", unsafe_allow_html=True)
                if st.button(f"📌 选择此岗位", key=f"select_jd_{i}"):
                    st.session_state.jd_text = snippet
                    st.session_state.selected_jd = jd
                    st.rerun()
    else:
        st.info("未找到相关岗位，尝试修改搜索关键词")
    st.divider()

# ── GitHub 项目预览 ──
if st.session_state.github_projects and not st.session_state.report:
    with st.expander("🐙 已导入的 GitHub 项目 (STAR 描述)", expanded=True):
        projects = st.session_state.github_projects.get("projects", [])
        for p in projects:
            st.markdown(f"""
            <div class="jd-card">
                <strong style="color:#F0C060">{p.get('name','')}</strong>
                <span style="color:#6b7280;font-size:0.8rem;">{', '.join(p.get('tech_stack',[]))}</span>
                <p style="color:#d1d5db;font-size:0.85rem;margin-top:6px;">{p.get('star_description','')[:200]}</p>
            </div>""", unsafe_allow_html=True)
    st.divider()

# ── 触发分析 ──
if analyze_btn:
    if not st.session_state.resume_text:
        st.error("请先上传简历 PDF")
    elif not config.get_api_key():
        st.error("请设置 DEEPSEEK_API_KEY")
    else:
        st.session_state.analyzing = True
        st.session_state.thinking_log = []
        st.session_state.report = None
        st.rerun()

# ── 执行 Agent ──
if st.session_state.analyzing and not st.session_state.report:
    try:
        agent = ResumeAgent()
        jd = st.session_state.jd_text

        # 如果有 GitHub 项目，追加到简历文本中
        resume = st.session_state.resume_text
        if st.session_state.github_projects:
            star_texts = []
            for p in st.session_state.github_projects.get("projects", []):
                star_texts.append(
                    f"项目：{p.get('name','')} | 技术栈：{', '.join(p.get('tech_stack',[]))} | "
                    f"描述：{p.get('star_description','')}"
                )
            resume += "\n\n[GitHub 项目经历]\n" + "\n".join(star_texts)

        with st.status("🤔 Agent 启动...", expanded=True) as status:
            log_lines = []
            for step in agent.analyze(resume, jd_text=jd, session_id="streamlit"):
                if step["type"] == "tool_start":
                    msg = f"🔧 {step['tool']}"
                    log_lines.append(msg)
                    status.update(label=msg)
                elif step["type"] == "tool_end":
                    msg = f"✅ {step['tool']}"
                    log_lines.append(msg)
                    try:
                        data = json.loads(step.get("preview", ""))
                        if data.get("score"):
                            log_lines.append(f"   → {data.get('dimension','')}: {data['score']}/10")
                    except Exception:
                        pass
                elif step["type"] == "done":
                    st.session_state.report = step["report"]
                    status.update(label="✅ 分析完成", state="complete")
                else:
                    log_lines.append(f"💬 {step.get('content','')[:100]}")
            st.session_state.thinking_log = log_lines
        st.session_state.analyzing = False
        st.rerun()
    except Exception as e:
        st.error(f"分析出错: {e}")
        st.session_state.analyzing = False

# ── 渲染报告 ──
if st.session_state.report:
    report = st.session_state.report
    scores = report.get("scores", {})
    overall = report.get("overall_score", 0)

    st.markdown(f"""
    <div class="overall-card">
        <p class="overall-label">综合匹配评分</p>
        <p class="overall-score">{overall:.0f}<span style="font-size:1.5rem;color:#6b7280"> /100</span></p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if scores:
        cols = st.columns(4)
        for i, key in enumerate(["skill_match", "experience", "tech_depth", "ats_keywords"]):
            if key in scores:
                render_score_card(scores[key], cols[i])
        st.markdown("<br>", unsafe_allow_html=True)
        if "overall_impression" in scores:
            cl, _, _ = st.columns([1, 1, 1])
            render_score_card(scores["overall_impression"], cl)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📝 修改建议")
    all_sug = []
    for dk, dd in scores.items():
        for s in dd.get("suggestions", []):
            all_sug.append({"dim": dd.get("name", dk), "text": s})
    for i, s in enumerate(all_sug):
        cls = "suggestion-high" if i == 0 else ("suggestion-mid" if i < 3 else "suggestion-low")
        ind = ["🔴", "🟡", "🟢"][min(i, 2)]
        st.markdown(f"""<div class="suggestion-item {cls}"><strong style="color:#F0C060">[{s['dim']}]</strong> {ind} {s['text']}</div>""", unsafe_allow_html=True)

    st.divider()
    if report.get("summary"):
        st.markdown("### 📊 分析总结")
        st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:20px;color:#d1d5db;line-height:1.7;">{report['summary']}</div>""", unsafe_allow_html=True)

    if st.session_state.thinking_log:
        with st.expander("🧠 Agent 思考过程", expanded=False):
            for line in st.session_state.thinking_log:
                st.caption(line)

# ── 空状态 ──
elif not st.session_state.report and not st.session_state.analyzing and not st.session_state.searched_jds:
    cl, cc, cr = st.columns([1, 2, 1])
    with cc:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <p style="font-size:4rem;margin:0">📋</p>
            <p style="color:#6b7280;font-size:1.1rem;margin-top:16px;">
                上传简历 PDF<br>可选：粘贴 JD / 搜索岗位 / 导入 GitHub
            </p>
            <p style="color:#4b5563;font-size:0.82rem;">
                10 Tools · RAG 检索 · SQLite Memory · 5 维评分
            </p>
        </div>""", unsafe_allow_html=True)
