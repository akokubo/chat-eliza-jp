import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnableLambda
from eliza.eliza import Eliza

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
    st.session_state.messages.append(AIMessage(content=initial_msg))

# ===============================
# LangChain Expression Languageを使ったチェインの定義
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
# ユーザー入力の受付と処理（LCEL使用）
# ===============================
user_input = st.chat_input("ELIZAに質問してみてください")
if user_input:
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(user_input)

    # LCELを使用してELIZAの応答を生成
    response = eliza_chain.invoke(user_input)

    ai_msg = AIMessage(content=response)
    st.session_state.messages.append(ai_msg)
    with st.chat_message("assistant", avatar=":material/psychology:"):
        st.markdown(response)
