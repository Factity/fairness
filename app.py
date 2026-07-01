

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
<html>
<head>
    <meta charset="utf-8">
    <title>GPT Token Visualizer</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; }
        textarea { width: 100%; font-size: 1.2rem; padding: 0.5rem; }
        button { margin-top: 0.5rem; font-size: 1rem; padding: 0.4em 1em; }
        .token-container { margin-top: 2rem; line-height: 2.2; font-size: 1.2rem; }
        .token {
            padding: 2px 0; border-radius: 3px; margin: 1px;
            display: inline-block; white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>🔠 GPT Token Visualizer</h1>
    <form method="POST">
        <textarea name="text" rows="3" placeholder="Type something...">{{ input_text }}</textarea><br>
        <button type="submit">Visualize</button>
    </form>
    {% if tokens_html %}
    <div class="token-container">{{ tokens_html | safe }}</div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    tokens_html = ""
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        if input_text:
            encoded = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
            token_ids = encoded["input_ids"][0].tolist()
            spans = []
            for tid in token_ids:
                color_info = get_token_color(tid)
                token_str = tokenizer.decode([tid]).replace("\u0120", " ")
                token_str = token_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                token_str = token_str.replace(" ", "&nbsp;")
                span = (
                    f'<span class="token" style="background-color:{color_info["bg"]};'
                    f'color:{color_info["fg"]};" title="Token ID: {tid}">{token_str}</span>'
                )
                spans.append(span)
            tokens_html = "".join(spans)
    return render_template_string(HTML_TEMPLATE, input_text=input_text, tokens_html=tokens_html)


app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)

