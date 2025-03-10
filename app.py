import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from eliza.eliza import Eliza

# ===============================
# 翻訳に使うLLMの設定
# ===============================
llm = ChatOpenAI(
    model_name="lucas2024/gemma-2-2b-jpn-it:q8_0",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key='ollama',
    temperature=0.5,
)

def translate_to_english(text):
    """日本語を英語に翻訳"""
    return llm.invoke(f"あなたは優秀な翻訳家です。次の文は精神科医との面談で患者が発した言葉です。これを英語に翻訳して、英語だけをシンプルに返して: {text}").content

def translate_to_japanese(text):
    """英語を日本語に翻訳"""
    return llm.invoke(f"あなたは優秀な翻訳家です。次の文は患者との面談で精神科医が発した言葉です。これを日本語に翻訳して、日本語だけをシンプルに返して: {text}").content

# ===============================
# ページの設定
# ===============================
# Streamlitのページタイトル、アイコン、レイアウトを設定しています。
st.set_page_config(
    page_title="ELIZAチャットボット",         # ブラウザタブに表示されるタイトル
    page_icon=":material/psychology:",         # ページアイコン（絵文字で指定）
    layout="wide"                              # ページのレイアウトを横幅いっぱいに広げる
)
st.title("ELIZAチャットボット")  # ページの見出しを表示

# ===============================
# セッション状態の初期化
# ===============================
# セッション状態を利用して、チャットの履歴やELIZAボットのインスタンスを保持します。

# チャットメッセージのリストが未定義の場合、空のリストとして初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# ELIZAボットのインスタンスが未定義の場合、生成と初期設定を行う
if "eliza_bot" not in st.session_state:
    st.session_state.eliza_bot = Eliza()               # ELIZAボットのインスタンスを生成
    st.session_state.eliza_bot.load("eliza/doctor.txt")  # スクリプトファイル（例: 医者の会話スクリプト）をロード
    initial_msg = st.session_state.eliza_bot.initial()   # 初期メッセージを取得
    st.session_state.messages.append(AIMessage(content=translate_to_japanese(initial_msg)))  # 初期メッセージをメッセージ履歴に追加

# ===============================
# ELIZAボットへの問い合わせ処理をラップ
# ===============================
# ユーザーからの入力に対して、ELIZAボットの応答を生成する関数を定義します。
eliza_chain = RunnableLambda(lambda x: st.session_state.eliza_bot.respond(x) or st.session_state.eliza_bot.final())

# ===============================
# これまでのチャットメッセージを画面に表示
# ===============================
# セッション状態に保存された全てのメッセージをループで表示します。
for message in st.session_state.messages:
    # メッセージの種類（ユーザー or アシスタント）を判定
    role = "assistant" if isinstance(message, AIMessage) else "user"
    # それぞれに対応するアバター（アイコン）を設定
    avatar = ":material/psychology:" if role == "assistant" else ":material/person:"
    # st.chat_messageを用いて、チャット風のUIにメッセージを表示
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.content)

# ===============================
# ユーザーからの新規入力受付
# ===============================
# チャット入力欄を表示し、ユーザーの入力を待ちます。
user_input = st.chat_input("ELIZAに質問してみてください")
if user_input:
    # ユーザーの入力をHumanMessageとして作成し、履歴に追加
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)
    # ユーザーのメッセージをチャットUIに表示
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(user_input)

    # ELIZAボットにユーザーの入力を英語で渡して応答を生成。応答を日本語に翻訳
    response = translate_to_japanese(eliza_chain.invoke(translate_to_english(user_input)))

    # 応答をAIMessageとして作成し、履歴に追加
    ai_msg = AIMessage(content=response)
    st.session_state.messages.append(ai_msg)
    # ボットの応答をチャットUIに表示
    with st.chat_message("assistant", avatar=":material/psychology:"):
        st.markdown(response)
