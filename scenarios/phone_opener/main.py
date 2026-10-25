"""Phone opener - AI-generated opening lines for phone calls using dynamic template grammar."""

import json
import re

import anthropic
import gradio as gr
from anthropic.types import TextBlock

SYSTEM_PROMPT = """あなたは電話コミュニケーションの専門家です。
日本のビジネス・日常コミュニケーションにおける「配慮表現」に深く精通しています。

以下の動的テンプレート文法に従って、電話の最初の一言を生成してください：

[時間配慮] + [関係性シグナル] + [用件ヒント] + [許可リクエスト]

各変数の意味：
- 時間配慮: 相手の時間・状況への配慮を示す表現
- 関係性シグナル: 発信者と受信者の関係を適切に示す表現
- 用件ヒント: 通話の目的を簡潔に予告する表現
- 許可リクエスト: 相手の同意・準備を確認する表現

出力形式は必ず以下のJSON形式にしてください：
{
  "variants": [
    {
      "text": "完全な最初の一言のテキスト",
      "breakdown": {
        "時間配慮": "該当する部分",
        "関係性シグナル": "該当する部分",
        "用件ヒント": "該当する部分",
        "許可リクエスト": "該当する部分"
      },
      "tone": "トーンの説明（丁寧・フレンドリー等）"
    }
  ],
  "caller_tip": "電話をかける前の一言アドバイス（発話者の心構え）"
}

バリエーションは3つ生成し、トーンや丁寧さのレベルを少しずつ変えてください。"""


def generate_opener(relationship: str, purpose: str, timing: str, urgency: str, prior_contact: str) -> str:
    client = anthropic.Anthropic()

    user_message = f"""以下の状況で電話をかけます。最初の一言テンプレートを3パターン生成してください。

【電話の状況】
- 相手との関係性: {relationship}
- 連絡の用件: {purpose}
- 電話する時間帯: {timing}
- 緊急度: {urgency}
- 事前連絡: {prior_contact}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    block = response.content[0]
    if not isinstance(block, TextBlock):
        return "エラー: テキスト応答が得られませんでした。"
    raw = block.text.strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return raw

    data = json.loads(json_match.group())
    return _format_output(data)


def _format_output(data: dict) -> str:  # type: ignore[type-arg]
    lines = []

    for i, v in enumerate(data.get("variants", []), 1):
        lines.append(f"## パターン {i}  ─  {v.get('tone', '')}")
        lines.append(f"\n**「{v['text']}」**\n")

        breakdown = v.get("breakdown", {})
        if breakdown:
            lines.append("| 変数 | 内容 |")
            lines.append("|------|------|")
            for key, val in breakdown.items():
                lines.append(f"| {key} | {val} |")
        lines.append("")

    tip = data.get("caller_tip", "")
    if tip:
        lines.append("---")
        lines.append(f"**電話前のコツ:** {tip}")

    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="電話の最初の一言ジェネレーター") as demo:
        gr.Markdown(
            """# 電話の最初の一言ジェネレーター
電話前に、相手への配慮表現を含んだ「最初の一言」をAIが生成します。

_動的テンプレート文法: [時間配慮] + [関係性シグナル] + [用件ヒント] + [許可リクエスト]_
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 状況を入力")

                relationship = gr.Dropdown(
                    choices=["初対面", "知人・同僚", "上司・先輩", "部下・後輩", "取引先", "友人・家族"],
                    label="相手との関係性",
                    value="取引先",
                )
                purpose = gr.Textbox(
                    label="連絡の用件",
                    placeholder="例：先日お送りした提案書についての確認",
                    lines=2,
                )
                timing = gr.Dropdown(
                    choices=["午前（9〜12時）", "昼（12〜13時）", "午後（13〜18時）", "夕方（18〜20時）"],
                    label="電話する時間帯",
                    value="午後（13〜18時）",
                )
                urgency = gr.Radio(
                    choices=["通常", "やや急ぎ", "緊急"],
                    label="緊急度",
                    value="通常",
                )
                prior_contact = gr.Radio(
                    choices=["突然の連絡", "事前に連絡すると伝えてある"],
                    label="事前連絡の有無",
                    value="突然の連絡",
                )

                generate_btn = gr.Button("最初の一言を生成", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 生成された最初の一言")
                output = gr.Markdown(value="_左のフォームを入力して「生成」ボタンを押してください_")

        generate_btn.click(
            fn=generate_opener,
            inputs=[relationship, purpose, timing, urgency, prior_contact],
            outputs=output,
        )

    return demo


def main() -> int:
    demo = build_ui()
    demo.launch(share=False)
    return 0
