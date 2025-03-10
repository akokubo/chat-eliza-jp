# ELIZAボット日本語版

## 使用したもの
 - Streamlit
 - LangChain
 - https://github.com/wadetb/eliza
 - [Ollama](https://ollama.com/)か[LM Studio](https://lmstudio.ai/)

## インストール
```
git clone https://github.com/akokubo/chat-eliza.git
cd chat-eliza-jp
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
git clone https://github.com/wadetb/eliza
```

## Ollamaの準備

## Ollamaの準備
1. Ollamaをインストール
   - Windowsの場合は、WSL2で仮想環境から `curl -fsSL https://ollama.com/install.sh | sh` でインストール
   - Macの場合は、[ダウンロード](https://ollama.com/download/windows)してインストール
2. Ollamaで大規模言語モデルの `lucas2024/gemma-2-2b-jpn-it:q8_0` をpullする。
```
ollama pull lucas2024/gemma-2-2b-jpn-it:q8_0
```
※大規模言語モデルは、自由に選べ、他のものでもいい。

※Ollamaの代わりに[LM Studio](https://lmstudio.ai/)も利用できる。その場合、「alfredplpl/gemma-2-2b-jpn-it」などのモデルをダウンロードする。LM Studioで、サーバーを走らせるには、左の「開発者」を選び、「Status」のトグルスイッチを切り替え「Running」にし、「Settings」で「ローカルネットワークでサービング」をオンにする。そして、app.pyの中の「BASE_URL」を `"http://localhost:1234/v1"`に変更する。右の「This model's API identifier」の値を「MODEL」に指定する。

※Windowsで、WSLからLM Studioに接続するには、ローカルネットワークでサービングをオンにし、右の「The local server is reachable at this address」のIPアドレスを `localhost` の代わりに指定する。

## 実行
最初に、プログラムを展開したフォルダに入る。
次に仮想環境に入っていない場合(コマンドプロンプトに(venv)と表示されていないとき)、仮想環境に入る。
```
source venv/bin/activate
```

Ollamaが起動していないかもしれないので、仮想環境に入っている状態で、大規模言語モデルのリストを表示する(すると起動していなければ、起動する)。
```
ollama list
```
※Ollamaの代わりにLM Studioを使っている場合は不要

仮想環境に入っている状態で、以下のコマンドでアプリを起動する。

```
python3 -m streamlit run app.py
```

![スクリーンショット](images/screenshot.jpg)

## 作者
[小久保 温(こくぼ・あつし)](https://akokubo.github.io/)

## ライセンス
[MIT License](LICENSE)
