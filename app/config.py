"""配置：常量与环境变量"""
import os

# ==================== 文档处理 ====================
ALLOWED_EXT = (".txt", ".md")

# ==================== 检索阈值 ====================
# Top-1 相似度低于此值时，认为库里没有相关内容，不调用模型
# 经验值，随语料规模和 embedding 模型调整
MIN_SCORE = 0.45

# ==================== 向量化 ====================
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# bge 系列在检索场景下，query 需要加中文指令前缀；文档片段不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# ==================== 大模型 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "你是一个严谨的技术知识库助手。\n"
    "请只依据【参考资料】回答问题，不要编造资料之外的内容。\n"
    "如果参考资料不足以回答，请明确回答「根据现有资料无法回答」。\n"
    "回答时用 [1][2] 的形式标注引用了哪几条资料。"
)