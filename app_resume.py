import os
import time
import json
import streamlit as st
from pdf_parser import extract_text_from_pdf
from resume_agent import ResumeAgent
import config_data as config


# ═══════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="Resume Analyzer — AI Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════
# 自定义 CSS — Editorial / Refined 深色主题
# ═══════════════════════════════════════════

st.markdown("""
<style>
/* ── 全局 ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #0f1117 0%, #1a1d28 50%, #151820 100%);
}

/* ── 标题 Hero ── */
.hero-title {
    font-family: 'DM Serif Display', 'Georgia', serif;
    font-size: 3.2rem;
    font-weight: 400;
    color: #F0C060;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0;
}

/* ── 综合评分卡 ── */
.overall-card {
    background: linear-gradient(135deg, rgba(240,192,96,0.12) 0%, rgba(240,192,96,0.04) 100%);
    border: 1px solid rgba(240,192,96,0.25);
    border-radius: 20px;
    padding: 32px 40px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.overall-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(240,192,96,0.12);
}
.overall-score {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    color: #F0C060;
    line-height: 1;
    margin: 0;
}
.overall-label {
    color: #9ca3af;
    font-size: 0.9rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 8px;
}

/* ── 维度评分卡片 ── */
.dim-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 24px;
    backdrop-filter: blur(8px);
    transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}
.dim-card:hover {
    transform: translateY(-2px);
    border-color: rgba(240,192,96,0.35);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.dim-name {
    font-size: 0.8rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.dim-score {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f1f5f9;
    line-height: 1;
}
.dim-weight {
    font-size: 0.75rem;
    color: #6b7280;
    margin-left: 6px;
}
.dim-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 6px;
    margin-top: 10px;
    overflow: hidden;
}
.dim-bar-fill {
    background: linear-gradient(90deg, #F0C060, #e8a830);
    border-radius: 6px;
    height: 100%;
    transition: width 0.6s ease;
}

/* ── 建议列表 ── */
.suggestion-high { border-left: 3px solid #ef4444; }
.suggestion-mid  { border-left: 3px solid #f59e0b; }
.suggestion-low  { border-left: 3px solid #10b981; }

.suggestion-item {
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.92rem;
    color: #d1d5db;
}

/* ── Agent 思考日志 ── */
.think-step {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    color: #6b7280;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background: rgba(15,17,23,0.8);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] h3 {
    color: #F0C060 !important;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── 按钮 ── */
div.stButton > button {
    background: linear-gradient(135deg, #F0C060 0%, #d4952a 100%) !important;
    color: #0f1117 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    width: 100% !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(240,192,96,0.25) !important;
}

/* ── 文件上传区域 ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.02) !important;
}

/* ── 分隔线 ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 24px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════

if "report" not in st.session_state:
    st.session_state.report = None
if "thinking_log" not in st.session_state:
    st.session_state.thinking_log = []
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""


def on_agent_step(step: dict):
    """Agent 思考步骤回调"""
    node = step.get("node", "")
    output = step.get("output", {})

    # 提取有意义的信息
    if node == "model":
        # 模型输出（可能含 tool_calls）
        msg = output.get("messages", [None])[0] if isinstance(output, dict) else None
        if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                st.session_state.thinking_log.append(f"🔧 调用工具: {tool_name}")
        elif msg and hasattr(msg, "content") and msg.content:
            pass  # 最终回复，不在这里显示

    elif node == "tools":
        for tool_msg in output.get("messages", []):
            tool_name = getattr(tool_msg, "name", "tool")
            st.session_state.thinking_log.append(f"✅ 工具完成: {tool_name}")
            # 尝试解析工具输出中是否包含评分
            try:
                content = tool_msg.content if hasattr(tool_msg, "content") else str(tool_msg)
                data = json.loads(content)
                if "score" in data:
                    dim = data.get("dimension", "")
                    score = data.get("score", 0)
                    st.session_state.thinking_log.append(f"   → {dim}: {score}/10")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass


# ═══════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════

def get_score_color(score: float) -> str:
    if score >= 8:
        return "#10b981"
    elif score >= 6:
        return "#f59e0b"
    else:
        return "#ef4444"


def get_suggestion_class(index: int, total: int) -> str:
    """建议的优先级标记"""
    if index == 0:
        return "suggestion-high"
    elif index < 3:
        return "suggestion-mid"
    return "suggestion-low"


def render_score_card(dim_key: str, dim_data: dict, col):
    """渲染单个维度评分卡片"""
    score = dim_data.get("score", 0)
    name = dim_data.get("name", dim_key)
    weight = dim_data.get("weight", 0)
    color = get_score_color(score)
    pct = int(score * 10)

    with col:
        st.markdown(f"""
        <div class="dim-card">
            <div class="dim-name">{name} <span class="dim-weight">× {int(weight*100)}%</span></div>
            <div class="dim-score" style="color:{color}">{score:.1f}<span style="font-size:1rem;color:#6b7280">/10</span></div>
            <div class="dim-bar-bg"><div class="dim-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}cc)"></div></div>
            <p style="color:#9ca3af;font-size:0.82rem;margin-top:10px;line-height:1.5;">{dim_data.get('analysis','')[:120]}</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📤 上传简历")
    uploaded_file = st.file_uploader(
        "支持 PDF 格式",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    st.markdown("### 📋 职位描述 (JD)")
    jd_input = st.text_area(
        "粘贴职位描述",
        height=200,
        placeholder="在此粘贴目标岗位的 JD 文本...",
        label_visibility="collapsed",
    )

    st.markdown("---")

    with st.expander("⚙️ 评分权重调整"):
        w_skill = st.slider("技能匹配度", 10, 50, int(config.scoring_weights["skill_match"] * 100), 5, format="%d%%")
        w_exp = st.slider("经验相关性", 10, 50, int(config.scoring_weights["experience"] * 100), 5, format="%d%%")
        w_depth = st.slider("技术深度", 10, 50, int(config.scoring_weights["tech_depth"] * 100), 5, format="%d%%")
        w_ats = st.slider("关键词/ATS", 5, 30, int(config.scoring_weights["ats_keywords"] * 100), 5, format="%d%%")
        w_impression = st.slider("综合印象", 5, 30, int(config.scoring_weights["overall_impression"] * 100), 5, format="%d%%")
        total_w = w_skill + w_exp + w_depth + w_ats + w_impression
        if total_w != 100:
            st.warning(f"权重合计 {total_w}%，应为 100%")

        # 更新运行时权重
        custom_weights = {
            "skill_match": w_skill / 100,
            "experience": w_exp / 100,
            "tech_depth": w_depth / 100,
            "ats_keywords": w_ats / 100,
            "overall_impression": w_impression / 100,
        }
        config.scoring_weights = custom_weights

    st.markdown("---")

    analyze_btn = st.button("🚀 开始分析", disabled=st.session_state.analyzing)

    st.markdown("---")
    st.markdown(
        '<p style="color:#6b7280;font-size:0.72rem;text-align:center">'
        'Powered by DeepSeek · Agent Architecture<br>'
        'Resume-to-JD Analysis System</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════
# 主区域
# ═══════════════════════════════════════════

st.markdown('<p class="hero-title">Resume Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI Agent · 简历与职位匹配分析</p>', unsafe_allow_html=True)
st.divider()

# ── 处理分析请求 ──
if analyze_btn:
    if uploaded_file is None:
        st.error("请先上传简历 PDF")
    elif not jd_input.strip():
        st.error("请输入职位描述 (JD)")
    elif not os.environ.get("DEEPSEEK_API_KEY"):
        st.error("请先设置环境变量 DEEPSEEK_API_KEY")
    else:
        st.session_state.analyzing = True
        st.session_state.thinking_log = []
        st.session_state.report = None

        # 解析 PDF
        try:
            file_bytes = uploaded_file.getvalue()
            st.session_state.resume_text = extract_text_from_pdf(file_bytes)
        except Exception as e:
            st.error(f"PDF 解析失败: {e}")
            st.session_state.analyzing = False

        st.session_state.jd_text = jd_input.strip()

# ── 执行 Agent 分析 ──
if st.session_state.analyzing and not st.session_state.report:
    try:
        agent = ResumeAgent(on_step=on_agent_step)

        with st.spinner(""):
            # 思考日志实时展示区域
            log_placeholder = st.empty()

            # 执行分析
            report = agent.analyze(
                st.session_state.resume_text,
                st.session_state.jd_text,
            )
            st.session_state.report = report
            st.session_state.analyzing = False

            # 最后一次刷新日志
            log_placeholder.empty()
            st.rerun()

    except Exception as e:
        st.error(f"分析出错: {e}")
        st.session_state.analyzing = False

# ── 渲染结果 ──
if st.session_state.report:
    report = st.session_state.report
    scores = report.get("scores", {})

    # ── 综合评分 ──
    overall = report.get("overall_score", 0)
    st.markdown(f"""
    <div class="overall-card">
        <p class="overall-label">综合匹配评分</p>
        <p class="overall-score">{overall:.0f}<span style="font-size:1.5rem;color:#6b7280"> /100</span></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 5 维度卡片（2+2+1 布局） ──
    if scores:
        dim_keys = ["skill_match", "experience", "tech_depth", "ats_keywords", "overall_impression"]

        # 第一行：4 列
        row1_keys = dim_keys[:4]
        cols = st.columns(4)
        for i, key in enumerate(row1_keys):
            if key in scores:
                render_score_card(key, scores[key], cols[i])

        # 第二行：综合印象单独一列 + 空列
        st.markdown("<br>", unsafe_allow_html=True)
        if "overall_impression" in scores:
            col_l, col_m, col_r = st.columns([1, 1, 1])
            render_score_card("overall_impression", scores["overall_impression"], col_l)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── 修改建议 ──
    st.markdown("### 📝 修改建议")
    all_suggestions = []
    for dim_key, dim_data in scores.items():
        for s in dim_data.get("suggestions", []):
            all_suggestions.append({
                "dimension": dim_data.get("name", dim_key),
                "text": s,
            })

    if all_suggestions:
        for i, s in enumerate(all_suggestions):
            css_class = get_suggestion_class(i, len(all_suggestions))
            indicator = ["🔴", "🟡", "🟢"][min(i, 2)]
            st.markdown(f"""
            <div class="suggestion-item {css_class}">
                <strong style="color:#F0C060">[{s['dimension']}]</strong> {indicator} {s['text']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无具体建议")

    st.divider()

    # ── 综合总结 ──
    summary = report.get("summary", "")
    if summary:
        st.markdown("### 📊 分析总结")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:20px 24px;color:#d1d5db;line-height:1.7;">
        {summary}
        </div>
        """, unsafe_allow_html=True)

    # ── Agent 思考日志 ──
    if st.session_state.thinking_log:
        with st.expander("🧠 Agent 思考过程", expanded=False):
            for step in st.session_state.thinking_log:
                st.markdown(f'<div class="think-step">{step}</div>', unsafe_allow_html=True)


# ── 初始空状态 ──
elif not st.session_state.report and not st.session_state.analyzing:
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <p style="font-size:4rem;margin:0">📋</p>
            <p style="color:#6b7280;font-size:1.1rem;margin-top:16px;">
                上传简历 PDF，粘贴职位描述<br>
                让 AI Agent 为您分析匹配度
            </p>
            <p style="color:#4b5563;font-size:0.82rem;">
                支持：技能匹配 · 经验相关性 · 技术深度 · ATS关键词 · 综合印象
            </p>
        </div>
        """, unsafe_allow_html=True)
