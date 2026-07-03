import threading
import time
from flask import Flask, render_template_string, request
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA

# ---------- 1. Build the colour map ----------
print("⏳ Loading GPT-2 model (first run downloads ~500 MB)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
model.eval()

embed_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()
pca = PCA(n_components=3)
pca.fit(embed_matrix)
transformed = pca.transform(embed_matrix)
mins = transformed.min(axis=0)
maxs = transformed.max(axis=0)

color_cache = {}
for tid in range(embed_matrix.shape[0]):
    vec = embed_matrix[tid]
    pca_vec = pca.transform(vec.reshape(1, -1))[0]
    rgb01 = (pca_vec - mins) / (maxs - mins + 1e-8)
    rgb = (rgb01 * 255).astype(int)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    text_color = "black" if luminance > 128 else "white"
    color_cache[tid] = {"bg": f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", "fg": text_color}
print("✅ Colour map ready!")


def get_token_color(token_id):
    return color_cache.get(token_id, {"bg": "#000", "fg": "white"})


# ---------- 2. Flask app ----------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GPT Token Visualizer</title>
    <style>
        :root {
            --bg: #f5f7fa;
            --card-bg: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #e5e7eb;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --radius: 12px;
            --shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.02);
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
            --font-mono: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            display: flex;
            justify-content: center;
            padding: 2rem 1rem;
            min-height: 100vh;
        }

        .container {
            max-width: 900px;
            width: 100%;
        }

        .card {
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        textarea {
            width: 100%;
            font-size: 1.1rem;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #fafbfc;
            resize: vertical;
            font-family: var(--font-mono);
            line-height: 1.5;
            transition: border 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
        }

        button {
            margin-top: 1rem;
            font-size: 1rem;
            font-weight: 600;
            padding: 0.6em 1.8em;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }

        button:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(0);
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #374151;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }

        .token-container {
            line-height: 2.4;
            word-break: break-word;
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            background: #f9fafb;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
        }

        .token {
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
            white-space: nowrap;
            font-family: var(--font-mono);
            font-size: 1.05rem;
            font-weight: 500;
            transition: transform 0.1s, box-shadow 0.1s;
        }

        .token:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 2;
        }

        .bpe-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }

        .bpe-table th {
            text-align: left;
            padding: 0.6rem 0.75rem;
            background: #f3f4f6;
            font-weight: 600;
            color: #374151;
            border-bottom: 2px solid var(--border);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .bpe-table td {
            padding: 0.6rem 0.75rem;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }

        .bpe-table tr:last-child td {
            border-bottom: none;
        }

        .bpe-table tr:hover td {
            background: #f0f4ff;
        }

        .color-swatch {
            display: inline-block;
            width: 18px;
            height: 18px;
            border-radius: 3px;
            margin-right: 8px;
            vertical-align: middle;
            border: 1px solid rgba(0,0,0,0.1);
        }

        .hex-bytes {
            font-family: var(--font-mono);
            background: #f1f5f9;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #334155;
        }

        .empty-message {
            color: var(--muted);
            font-style: italic;
            padding: 1rem 0;
        }

        @media (max-width: 600px) {
            body { padding: 1rem; }
            .card { padding: 1.25rem; }
            .bpe-table { font-size: 0.8rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🔠 GPT Token Visualizer</h1>
            <form method="POST">
                <textarea name="text" rows="3" placeholder="Type something...">{{ input_text }}</textarea><br>
                <button type="submit">Visualize</button>
            </form>
        </div>

        {% if tokens_html %}
        <div class="card">
            <div class="section-title">🧩 Tokenized Output (coloured by embedding PCA)</div>
            <div class="token-container">{{ tokens_html | safe }}</div>
        </div>

        <div class="card">
            <div class="section-title">📋 Byte‑Pair Encoding Details</div>
            {% if bpe_info %}
            <table class="bpe-table">
                <thead>
                    <tr>
                        <th>Token ID</th>
                        <th>Decoded Text</th>
                        <th>Byte Encoding (hex)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for tok in bpe_info %}
                    <tr>
                        <td>{{ tok.id }}</td>
                        <td>
                            <span class="color-swatch" style="background-color: {{ tok.bg }};"></span>
                            <code>{{ tok.text }}</code>
                        </td>
                        <td><span class="hex-bytes">{{ tok.bytes_hex }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="empty-message">No tokens to display yet.</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    tokens_html = ""
    input_text = ""
    bpe_info = []  # will hold token details

    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        if input_text:
            encoded = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
            token_ids = encoded["input_ids"][0].tolist()

            spans = []
            for tid in token_ids:
                # Decode token string (escape HTML entities)
                token_str = tokenizer.decode([tid])
                # Keep original for hex encoding before replacing spaces
                raw_str = token_str.replace("\u0120", " ")  # Ġ -> space
                # Escape HTML special chars
                escaped = raw_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                # Convert space to &nbsp; for visual token display
                display_str = escaped.replace(" ", "&nbsp;")

                color_info = get_token_color(tid)
                span = (
                    f'<span class="token" style="background-color:{color_info["bg"]};'
                    f'color:{color_info["fg"]};" title="Token ID: {tid}">{display_str}</span>'
                )
                spans.append(span)

                # Build BPE info
                # Raw token text (with real spaces for table)
                token_text = raw_str  # already replaced Ġ with space
                # Get UTF-8 bytes and format as hex
                token_bytes = token_text.encode("utf-8")
                hex_str = " ".join(f"{b:02x}" for b in token_bytes)
                bpe_info.append({
                    "id": tid,
                    "text": token_text,  # may contain spaces, will be shown in <code>
                    "bytes_hex": hex_str,
                    "bg": color_info["bg"]  # for colour swatch
                })

            tokens_html = "".join(spans)

    return render_template_string(
        HTML_TEMPLATE,
        input_text=input_text,
        tokens_html=tokens_html,
        bpe_info=bpe_info
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)