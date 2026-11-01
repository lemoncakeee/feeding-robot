# feeding-robot

> 本リポジトリは、Physical AI 応用編1 講座 (松尾・岩澤研究室) の最終課題として作成したものです。

手が不自由な方を想定した、音声対話型の食事支援ロボットのシミュレーションです。
自由な日本語の発話から動作計画を作り、7軸ロボットアーム Franka Emika Panda がスプーンで口元まで食事を運びます。

## 特徴

- 自由な発話から、LLM が動作計画を生成・変更します
- 手首カメラで口の位置を追跡し、頭が動いてもスプーンの目標を補正します
- 返答は、固定した声で事前生成した音声を再生します

## 使用技術

| 役割 | 使用技術 |
|---|---|
| シミュレーター | PyBullet、Franka Emika Panda（7軸アーム） |
| 発話検出 | Silero VAD |
| 音声認識 | speech.cpp（Haas-ja-v1.0） |
| 動作計画 | Ollama（qwen3:4b-instruct） |
| 口の検出 | MediaPipe Face Mesh、色による補助検出 |
| 音声合成 | omnivoice.cpp |

アームの軌道は学習モデルではなく、経由点と逆運動学で制御しています。

## セットアップ

macOS（Apple Silicon）と [uv](https://docs.astral.sh/uv/) が必要です。

```sh
CFLAGS="-Dfdopen=fdopen" CPPFLAGS="-Dfdopen=fdopen" uv sync
ollama pull qwen3:4b-instruct
```

`pybullet` はソースからビルドされるため、上記のflagが必要です。

あわせて、次の2つをインストールしてください。

- [speech.cpp](https://github.com/nyosegawa/speech.cpp)
- [omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp)：`~/.local/share/omnivoice.cpp` にビルドし、`omnivoice-base-Q8_0.gguf` と `omnivoice-tokenizer-F32.gguf` を `models/` に配置します

## 実行

Ollama を起動した状態で実行します。

```sh
uv run python main.py
```

「ご飯ちょうだい」「待って」「いいよ」「ご飯と味噌汁を交互に2口ずつ」「やっぱりプリンにして」のように話しかけてください。

| オプション | 内容 |
|---|---|
| `--no-mic` | マイクを使わず、文字入力で操作する |
| `--still-head` | 頭の動きと口の追跡を無効にする |

実行中は `w` で一時停止、`r` で再開、`q` で終了します。

## 構成

```
main.py                 起動
scene.yaml              料理とシーンの定義
assets/voice/           音声クローン用の参照音声
feeding_robot/
  speech/               発話検出、音声認識、音声合成
  planning/             動作計画、LLM、安全性の検証
  simulation/           シーン、アーム制御、口の追跡
tests/
```

## テスト

```sh
uv run pytest         # Ollama やマイクは不要
uv run pytest --live  # 起動中の Ollama を使う
```

## 制限事項

- シミュレーションのみで、実機では動作確認していません
- スプーンと食べ物はvisualだけで、物理的な接触はありません
- 料理の位置は `scene.yaml` の固定値で、カメラでは検出しません
- 口元に近い距離の検出は色を使っており、シミュレーション上の顔に合わせて調整しています
