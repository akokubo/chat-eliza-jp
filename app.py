import streamlit as st
import re
import random
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
# elizaディレクトリにあるeliza.pyを読み込みます
from eliza.eliza import Eliza

# ==========================================
# 1. 設定: LLM (Ollama) の接続設定
# ==========================================
# ★ 翻訳機能を有効にするにはここを True にしてください
USE_LLM = True

# 利用するOllamaのモデル名（環境に合わせて書き換えてください: gemma2, llama3, phi3 等）
OLLAMA_MODEL = "gemma3:4b-it-qat" 

# LLMの初期化
llm = None
if USE_LLM:
    try:
        llm = ChatOpenAI(
            model_name=OLLAMA_MODEL,
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            temperature=0.0,
            request_timeout=10
        )
    except Exception as e:
        st.warning(f"LLMの初期化に失敗しました: {e}")
        USE_LLM = False

# ==========================================
# 2. 翻訳関数
# ==========================================
def translate(text, target_lang="English"):
    """
    LLMを使って翻訳を行う関数。
    LLMがオフの場合やエラー時は、モック（模擬）テキストを返す。
    """
    if not USE_LLM or not llm:
        if target_lang == "English":
            return text # 英語への翻訳失敗時はそのまま返す（ELIZAは動かなくなるがエラーは防ぐ）
        else:
            return f"(翻訳不可) {text}"

    if target_lang == "English":
        prompt = f"Translate the following Japanese text into natural, simple English conversational text for a therapy session. Output ONLY the translation.\n\nJapanese: {text}"
    else:
        prompt = f"Translate the following English response from a psychotherapist (ELIZA) into natural, polite Japanese (desu/masu style). Output ONLY the translation.\n\nEnglish: {text}"

    try:
        # invokeで実行
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"[Translation Error] {text}"

def is_japanese(text):
    return bool(re.search(r'[ぁ-んァ-ン一-龯]', text))

# ==========================================
# 3. 内部ロジック可視化用 ELIZA拡張クラス
# ==========================================
class TraceableEliza(Eliza):
    def __init__(self):
        super().__init__()
        self.trace_log = {
            "key": None,
            "decomp": None,
            "reasmb": None,
            "source": None,       # rule / memory / xnone
            "saved_memory": None  # 今回保存された記憶
        }

    def reset_trace(self):
        self.trace_log = {
            "key": None, "decomp": None, "reasmb": None, 
            "source": None, "saved_memory": None
        }

    def respond(self, text):
        """
        ELIZAのrespondメソッドをオーバーライドし、思考プロセスを記録する。
        """
        self.reset_trace()

        # --- 前処理 (句読点処理) ---
        text = text.lower()
        text = re.sub(r'\s*\.+\s*', ' . ', text)
        text = re.sub(r'\s*,+\s*', ' , ', text)
        text = re.sub(r'\s*;+\s*', ' ; ', text)

        words = [w for w in text.split(' ') if w]
        words = self._sub(words, self.pres)
        
        # キーワードの取得とソート
        keys = [self.keys[w] for w in words if w in self.keys]
        keys = sorted(keys, key=lambda k: -k.weight)

        output = None

        # --- キーワードマッチング ---
        for key in keys:
            output = self._match_key_traceable(words, key)
            if output:
                break
        
        # --- 記憶 (Memory) の利用 ---
        if not output:
            if self.memory:
                index = random.randrange(len(self.memory))
                output = self.memory.pop(index)
                
                self.trace_log["source"] = "memory"
                self.trace_log["key"] = "Memory Stack (Recall)"
                self.trace_log["decomp"] = "N/A"
                self.trace_log["reasmb"] = " ".join(output)
            else:
                # --- XNONE (理解不能) ---
                output = self._next_reasmb(self.keys['xnone'].decomps[0])
                
                self.trace_log["source"] = "xnone"
                self.trace_log["key"] = "xnone"
                self.trace_log["decomp"] = "N/A"
                self.trace_log["reasmb"] = " ".join(output)

        return " ".join(output)

    def _match_key_traceable(self, words, key):
        """
        _match_key を拡張し、詳細なマッチング情報を記録する。
        """
        for decomp in key.decomps:
            results = self._match_decomp(decomp.parts, words)
            if results is None:
                continue
            
            # マッチ成功後の処理
            results = [self._sub(words, self.posts) for words in results]
            reasmb = self._next_reasmb(decomp)

            # goto処理
            if reasmb[0] == 'goto':
                goto_key = reasmb[1]
                if goto_key in self.keys:
                    return self._match_key_traceable(words, self.keys[goto_key])
                return None

            output = self._reassemble(reasmb, results)

            # ★重要★ 記憶への保存 (save=True) の場合
            if decomp.save:
                self.memory.append(output)
                # ログに記録するが、リターンせずに探索を続ける (continue)
                self.trace_log["saved_memory"] = " ".join(output)
                continue
            
            # 通常の応答生成
            self.trace_log["source"] = "rule"
            self.trace_log["key"] = key.word
            self.trace_log["decomp"] = " ".join(decomp.parts)
            self.trace_log["reasmb"] = " ".join(reasmb)
            return output
            
        return None

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="ELIZA Bilingual Debugger", layout="wide", page_icon="🧠")
st.title("🧠 ELIZA Bilingual Debugger")

# --- セッション状態の初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "eliza_bot" not in st.session_state:
    try:
        bot = TraceableEliza()
        bot.load("eliza/doctor.txt")
        st.session_state.eliza_bot = bot
        
        # 初回メッセージ
        init_en = bot.initial()
        init_jp = translate(init_en, "Japanese")
        st.session_state.messages.append({
            "role": "assistant",
            "content_jp": init_jp,
            "content_en": init_en,
            "trace": {"source": "initial"}
        })
    except FileNotFoundError:
        st.error("エラー: `eliza/doctor.txt` または `eliza/eliza.py` が見つかりません。")
        st.stop()

# --- サイドバー: 記憶スタックの可視化 ---
with st.sidebar:
    st.header("💾 Memory Stack")
    st.markdown("会話の中で「後で役に立つ」と判断された情報がここに蓄積されます。")
    
    if st.session_state.eliza_bot.memory:
        # スタックのように上を最新にするため reversed を使用
        for i, mem in enumerate(reversed(st.session_state.eliza_bot.memory)):
            mem_str = " ".join(mem) if isinstance(mem, list) else str(mem)
            st.code(f"[{len(st.session_state.eliza_bot.memory) - i}] {mem_str}", language="text")
    else:
        st.info("📭 **現在は空です**")
        with st.expander("💡 どうすれば記憶される？", expanded=True):
            st.caption("""
            ELIZAは特定のキーワード（`my` 家族、`my` ペットなど）に反応して記憶を作ります。
            
            **試してみよう:**
            - 「私の母は料理が得意です」
            - 「私の彼は大阪出身です」
            """)
    
    st.divider()
    st.markdown("### 凡例")
    st.markdown("**Key**: ヒットしたキーワード")
    st.markdown("**Decomp**: 入力分解パターン (* はワイルドカード)")
    st.markdown("**Reasmb**: 選ばれた返答テンプレート")

# --- メインチャット画面 ---
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = ":material/psychology:" if role == "assistant" else ":material/person:"
    
    with st.chat_message(role, avatar=avatar):
        # 1. メインのテキスト表示（日本語）
        st.markdown(f"**{msg['content_jp']}**")
        
        # 2. 翻訳元の英語を表示
        if role == "user" and msg['content_jp'] != msg['content_en']:
            st.caption(f"🇬🇧 English Input: `{msg['content_en']}`")
        elif role == "assistant":
            # ELIZAの場合は詳細情報を表示
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🇬🇧 ELIZA Thought: `{msg['content_en']}`")
            
            # デバッグ情報の表示
            trace = msg.get("trace", {})
            source = trace.get("source")
            
            if source != "initial":
                with st.expander("🛠️ 内部ロジックを見る", expanded=False):
                    if source == "memory":
                        st.warning("🔄 **記憶スタックから放出** (Memory Recall)")
                    elif source == "xnone":
                        st.error("❓ **キーワードなし** (Fallback)")
                    else:
                        st.success(f"✅ **ルール適合** (Key: {trace.get('key')})")

                    st.text(f"Key   : {trace.get('key')}")
                    st.text(f"Decomp: {trace.get('decomp')}")
                    st.text(f"Reasmb: {trace.get('reasmb')}")
                    
                    if trace.get("saved_memory"):
                        st.info(f"📥 **記憶に保存しました:**\n{trace.get('saved_memory')}")

# --- 入力エリア ---
if prompt := st.chat_input("ここに日本語で入力してください (例: 母は厳しいです)"):
    # 1. ユーザー入力の処理
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(f"**{prompt}**")
        
        # 日本語なら翻訳、英語ならそのまま
        if is_japanese(prompt):
            input_en = translate(prompt, "English")
            st.caption(f"🇬🇧 English Input: `{input_en}`")
        else:
            input_en = prompt
            st.caption("🇬🇧 Direct Input")
            
    st.session_state.messages.append({
        "role": "user",
        "content_jp": prompt,
        "content_en": input_en
    })

    # 2. ELIZAの応答処理
    with st.chat_message("assistant", avatar=":material/psychology:"):
        with st.spinner("ELIZA is thinking..."):
            # ELIZA本体の処理
            response_en = st.session_state.eliza_bot.respond(input_en)
            
            # 内部状態のコピー
            current_trace = st.session_state.eliza_bot.trace_log.copy()
            
            # 英語 -> 日本語 翻訳
            response_jp = translate(response_en, "Japanese")
            
            # 表示のためのウェイト（演出）
            time.sleep(0.5)
            
            st.markdown(f"**{response_jp}**")
            st.caption(f"🇬🇧 ELIZA Thought: `{response_en}`")
            
            # メモリ保存があった場合にトーストで通知
            if current_trace.get("saved_memory"):
                st.toast(f"💾 Memory Saved: {current_trace['saved_memory']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content_jp": response_jp,
        "content_en": response_en,
        "trace": current_trace
    })
    
    # 状態更新のためリロード
    st.rerun()