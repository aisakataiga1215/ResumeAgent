
md5_path = "./md5.text"

# Chroma
collection_name="rag"
persist_directory="./chroma_db"

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

max_spliter_char_number= 1000  # 文本分割阈值

# 相似度K值
similarity_threshold =1     # 检索返回匹配的文档数量

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

# DeepSeek 模型配置（Resume Agent 使用）
deepseek_model_name = "deepseek-chat"
deepseek_api_base = "https://api.deepseek.com"

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}

# ===================== Resume Agent 配置 =====================
# 评分维度权重（总和为 1.0）
scoring_weights = {
    "skill_match": 0.30,
    "experience": 0.25,
    "tech_depth": 0.20,
    "ats_keywords": 0.15,
    "overall_impression": 0.10
}

# Agent 最大迭代次数
agent_max_iterations = 10

# Agent 系统提示词（作为工具调用前的唯一初始消息）
resume_agent_system_prompt = """你是一位资深 HR 和职业顾问，擅长分析简历与职位的匹配度。

你的分析流程：
1. 先用 extract_resume_info 提取简历中的技能、经验、项目、教育信息
2. 再用 extract_jd_requirements 提取 JD 中的关键要求
3. 然后用 analyze_dimension 逐个维度对比打分（技能匹配度、经验相关性、技术深度）
4. 用 check_ats_keywords 检查关键词覆盖情况
5. 最后用 generate_final_report 生成完整分析报告，额外包含综合印象维度的评分

每个维度都给出 1-10 分的评分和具体的修改建议。建议必须具体、可操作，包含原文问题和修改方向。"""