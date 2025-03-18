import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable
from eliza.eliza import Eliza

# ===============================
# Streamlitページの基本設定
# ===============================
st.set_page_config(
    page_title="ELIZAチャットボット日本語版",
    page_icon=":material/psychology:",
    layout="wide"
)
st.title("ELIZAチャットボット日本語版")

# ===============================
# LLMの設定（ローカルLLMを利用）
# ===============================
translator_llm = ChatOpenAI(
    model_name="gemma3",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key='ollama',
    temperature=0.5,
)

# ===============================
# ELIZAをRunnable化するクラス定義
# ===============================
class ElizaRunnable(Runnable):
    def __init__(self):
        self.eliza = Eliza()
        self.eliza.load("eliza/doctor.txt")
        self.initial_msg = self.eliza.initial()

    def invoke(self, input, config=None):
        if isinstance(input, dict):
            user_input = input.get('text', '').strip()
        else:
            user_input = str(input).strip()
        return self.eliza.respond(user_input) if user_input else self.initial_msg

# ===============================
# 翻訳プロンプトの最適化
# ===============================
translate_to_english_prompt = ChatPromptTemplate.from_template(
    "あなたは優秀な翻訳家です。次の文は精神科医との面談で患者が発した言葉です。"
    "これを英語に翻訳して、英語だけをシンプルに返して:\n\n{text}"
)

translate_to_japanese_prompt = ChatPromptTemplate.from_template(
    "あなたは優秀な翻訳家です。次の文は患者との面談で精神科医が発した言葉です。"
    "これを日本語に翻訳して、日本語だけをシンプルに返して:\n\n{text}"
)

# ===============================
# チェインの構築
# ===============================
first_chain = (
    ElizaRunnable()
    | translate_to_japanese_prompt
    | translator_llm
    | StrOutputParser()
)

chain = (
    translate_to_english_prompt
    | translator_llm
    | StrOutputParser()
    | ElizaRunnable()
    | translate_to_japanese_prompt
    | translator_llm
    | StrOutputParser()
)

# ===============================
# セッション状態の初期化
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_msg = first_chain.invoke({"text": ""})
    st.session_state.messages.append(AIMessage(content=initial_msg))

# ===============================
# メッセージ履歴の表示
# ===============================
for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    avatar = ":material/psychology:" if role == "assistant" else ":material/person:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.content)

# ===============================
# ユーザー入力の処理
# ===============================
user_input = st.chat_input("ELIZAに話しかけてみましょう", key="user_input")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(user_input)

    try:
        response = chain.invoke({"text": user_input})
    except ConnectionError:
        response = "⚠️ネットワークエラーが発生しました。接続を確認してください。"
    except Exception as e:
        response = f"⚠️エラーが発生しました: {str(e)}"

    st.session_state.messages.append(AIMessage(content=response))
    with st.chat_message("assistant", avatar=":material/psychology:"):
        st.markdown(response)
