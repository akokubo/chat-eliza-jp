import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from eliza.eliza import Eliza

llm = ChatOpenAI(
    model_name="lucas2024/gemma-2-2b-jpn-it:q8_0",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key='ollama',
    temperature=0.5,
)

def translate_to_english(text):
    """日本語を英語に翻訳"""
    return llm.invoke(f"あなたは優秀な翻訳家です。次の文を英語に翻訳して、英語だけをシンプルに返して: {text}").content

def translate_to_japanese(text):
    """英語を日本語に翻訳"""
    return llm.invoke(f"あなたは優秀な翻訳家です。次の文を日本語に翻訳して、日本語だけをシンプルに返して: {text}").content

# ===============================
# ページ設定
# ===============================
st.set_page_config(
    page_title="ELIZAチャットボット",
    page_icon=":material/psychology:",
    layout="wide"
)
st.title("ELIZAチャットボット")

# ===============================
# セッション状態の初期化
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "eliza_bot" not in st.session_state:
    st.session_state.eliza_bot = Eliza()
    st.session_state.eliza_bot.load("eliza/doctor.txt")
    initial_msg = st.session_state.eliza_bot.initial()
    st.session_state.messages.append(AIMessage(content=translate_to_japanese(initial_msg)))

# ===============================
# LangChainのRunnableLambdaを使ったチェインの定義
# ===============================
eliza_chain = RunnableLambda(lambda x: st.session_state.eliza_bot.respond(x) or st.session_state.eliza_bot.final())

# ===============================
# チャット履歴を画面に表示
# ===============================
for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    avatar = ":material/psychology:" if role == "assistant" else ":material/person:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.content)

# ===============================
# ユーザー入力の受付と処理（LangChainのRunnableLambdaを使用）
# ===============================
user_input = st.chat_input("ELIZAに質問してみてください")
if user_input:
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(user_input)

    # LangChainのRunnableLambdaを使用してELIZAの応答を生成
    response = translate_to_japanese(eliza_chain.invoke(translate_to_english(user_input)))

    ai_msg = AIMessage(content=response)
    st.session_state.messages.append(ai_msg)
    with st.chat_message("assistant", avatar=":material/psychology:"):
        st.markdown(response)
