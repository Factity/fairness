import hashlib
import colorsys
from flask import Flask, render_template_string, request
import regex
from transformers import GPT2TokenizerFast

# ---------- 1. Load GPT-2 tokenizer (fast, byte‑level BPE) ----------
print("⏳ Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print("✅ Tokenizer ready.")

# ---------- 2. GPT-2 regex pattern ----------
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
token_regex = regex.compile(GPT2_SPLIT_PATTERN)

# Sub‑pattern classification
RULE_PATTERNS = [
    ("contraction", regex.compile(r"^(?:'s|'t|'re|'ve|'m|'ll|'d)$")),
    ("letters",     regex.compile(r"^ ?\p{L}+$")),
    ("digits",      regex.compile(r"^ ?\p{N}+$")),
    ("other chars", regex.compile(r"^ ?[^\s\p{L}\p{N}]+$")),
    ("trailing ws", regex.compile(r"^\s+(?!\S)$")),
    ("whitespace",  regex.compile(r"^\s+$")),
]

def get_token_rule(token_str):
    for name, pat in RULE_PATTERNS:
        if pat.fullmatch(token_str):
            return name
    return "unknown"

def gpt2_regex_split(text):
    return [(m.group(), m.start(), m.end()) for m in token_regex.finditer(text)]

# ---------- 3. Colour helpers ----------
def hash_color(token: str):
    h = hashlib.sha256(token.encode("utf-8")).hexdigest()
    hue = int(h[:8], 16) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.7, 0.85)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = "black" if luminance > 128 else "white"
    return f"rgb({r},{g},{b})", text_color

# ---------- 4. Flask app ----------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GPT Full Tokenization Pipeline</title>
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
            --font-mono: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            display: flex;
            justify-content: center;
            padding: 2rem 1rem;
            min-height: 100vh;
        }
        .container { max-width: 1200px; width: 100%; }
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
        button:hover { background: var(--accent-hover); transform: translateY(-1px); }
        button:active { transform: translateY(0); }
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #374151;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }
        .regex-pattern {
            background: #1e293b;
            color: #e2e8f0;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            font-family: var(--font-mono);
            font-size: 0.95rem;
            overflow-x: auto;
            white-space: pre-wrap;
            margin-bottom: 1rem;
        }
        .regex-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 1rem;
        }
        .legend-item {
            font-size: 0.85rem;
            background: #f1f5f9;
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid #cbd5e1;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .step-container {
            line-height: 2.6;
            word-break: break-word;
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            background: #f9fafb;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            align-items: flex-start;
        }
        .step-token {
            display: inline-flex;
            align-items: center;
            white-space: nowrap;
            font-family: var(--font-mono);
            font-size: 1.05rem;
            font-weight: 500;
            border-radius: 6px;
            padding: 2px 6px 2px 2px;
            transition: transform 0.1s, box-shadow 0.1s;
        }
        .step-token:hover {
            transform: scale(1.03);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 20px;
            height: 20px;
            background: #1f2937;
            color: white;
            border-radius: 50%;
            font-size: 10px;
            font-weight: 700;
            margin-right: 4px;
            flex-shrink: 0;
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
        .bpe-table tr:last-child td { border-bottom: none; }
        .bpe-table tr:hover td { background: #f0f4ff; }
        .color-swatch {
            display: inline-block;
            width: 18px;
            height: 18px;
            border-radius: 3px;
            margin-right: 8px;
            vertical-align: middle;
            border: 1px solid rgba(0,0,0,0.1);
        }
        .rule-badge {
            display: inline-block;
            background: #e0e7ff;
            color: #3730a3;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
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
        .explanation {
            background: #f0f9ff;
            border-left: 4px solid #3b82f6;
            padding: 0.8rem 1rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            color: #1e3a8a;
            font-size: 0.95rem;
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
            <h1>🧠 GPT-2 Full Tokenization Pipeline</h1>
            <form method="POST">
                <textarea name="text" rows="3" placeholder="Type something...">{{ input_text }}</textarea><br>
                <button type="submit">Analyze</button>
            </form>
        </div>

        {% if input_text %}
        <!-- Stage 1: Regex pre‑tokenization -->
        <div class="card">
            <div class="section-title">🔍 Stage 1 – Regex Pre‑tokenization</div>
            <div class="explanation">
                GPT‑2 first splits the raw text into <strong>word‑like chunks</strong> using this regex.
                Whitespace is preserved and attached to the next token.
            </div>
            <div class="regex-pattern">{{ regex_pattern }}</div>
            <div class="regex-legend">
                <div class="legend-item"><span class="legend-dot" style="background:#bae6fd;"></span> contraction</div>
                <div class="legend-item"><span class="legend-dot" style="background:#bbf7d0;"></span> letters</div>
                <div class="legend-item"><span class="legend-dot" style="background:#fde68a;"></span> digits</div>
                <div class="legend-item"><span class="legend-dot" style="background:#fed7aa;"></span> other chars</div>
                <div class="legend-item"><span class="legend-dot" style="background:#e9d5ff;"></span> trailing ws</div>
                <div class="legend-item"><span class="legend-dot" style="background:#fecaca;"></span> whitespace</div>
            </div>
            <p style="margin-bottom: 0.5rem; font-weight: 600; font-size: 0.95rem;">
                Matched tokens:
            </p>
            <div class="step-container">{{ regex_tokens_html | safe }}</div>
            <table class="bpe-table" style="margin-top: 1.5rem;">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Chunk</th>
                        <th>Rule</th>
                        <th>UTF‑8 Bytes (hex)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for chunk in regex_info %}
                    <tr>
                        <td>{{ chunk.index }}</td>
                        <td><code>{{ chunk.text }}</code></td>
                        <td><span class="rule-badge">{{ chunk.rule }}</span></td>
                        <td><span class="hex-bytes">{{ chunk.bytes_hex }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Stage 2: Byte‑level BPE Encoding -->
        <div class="card">
            <div class="section-title">🔤 Stage 2 – Byte‑level BPE Encoding</div>
            <div class="explanation">
                Each chunk is represented as a sequence of <strong>UTF‑8 bytes</strong>. The tokenizer then applies
                byte‑pair merges to those byte sequences to produce <strong>subword tokens</strong>. The final
                token IDs are the model’s vocabulary indices.
            </div>
            <p style="font-weight: 600; margin-bottom: 0.5rem;">Full tokenization of the whole text:</p>
            <div class="step-container">
                {% for tok in bpe_tokens %}
                <span class="step-token" style="background-color:{{ tok.bg }}; color:{{ tok.fg }};"
                      title="ID: {{ tok.id }} – Byte(s): {{ tok.bytes_hex }}">
                    <span class="step-badge">{{ tok.id }}</span>
                    {{ tok.display }}
                </span>
                {% endfor %}
            </div>

            <table class="bpe-table" style="margin-top: 1.5rem;">
                <thead>
                    <tr>
                        <th>Token ID</th>
                        <th>Subword Text</th>
                        <th>Start / End</th>
                        <th>Raw Bytes (UTF‑8 hex)</th>
                        <th>Byte length</th>
                    </tr>
                </thead>
                <tbody>
                    {% for tok in bpe_tokens %}
                    <tr>
                        <td>{{ tok.id }}</td>
                        <td>
                            <span class="color-swatch" style="background-color: {{ tok.bg }};"></span>
                            <code>{{ tok.text }}</code>
                        </td>
                        <td>{{ tok.start }}–{{ tok.end }}</td>
                        <td><span class="hex-bytes">{{ tok.bytes_hex }}</span></td>
                        <td>{{ tok.byte_len }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Stage 3: Decoding -->
        <div class="card">
            <div class="section-title">↩️ Stage 3 – Decoding Back to Text</div>
            <div class="explanation">
                To recover the original text, the token IDs are converted back to their byte sequences,
                concatenated, and then decoded from UTF‑8. The reconstructed string is:
            </div>
            <div style="background: #f1f5f9; padding: 0.8rem; border-radius: 8px; font-family: var(--font-mono); font-size: 1.1rem; word-break: break-all;">
                “{{ decoded_text }}”
            </div>
            <p style="margin-top: 0.8rem; color: var(--muted); font-size: 0.9rem;">
                (Matches original: <strong>{{ decoded_text == input_text }}</strong>)
            </p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ---------- 5. Route ----------
@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    regex_tokens_html = ""
    regex_info = []
    bpe_tokens = []
    decoded_text = ""
    regex_pattern = GPT2_SPLIT_PATTERN

    if request.method == "POST":
        input_text = request.form.get("text", "")

        if input_text:
            # --- Stage 1: Regex splitting ---
            chunks = gpt2_regex_split(input_text)
            spans = []
            for idx, (token_str, start, end) in enumerate(chunks, start=1):
                bg, fg = hash_color(token_str)
                display = token_str.replace(" ", "·")
                escaped = display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                spans.append(
                    f'<span class="step-token" style="background-color:{bg}; color:{fg};" '
                    f'title="#{idx}: {token_str!r}">'
                    f'<span class="step-badge">{idx}</span>{escaped}</span>'
                )
                rule = get_token_rule(token_str)
                token_bytes = token_str.encode("utf-8")
                hex_str = " ".join(f"{b:02x}" for b in token_bytes)
                regex_info.append({
                    "index": idx,
                    "text": token_str,
                    "rule": rule,
                    "bytes_hex": hex_str
                })
            regex_tokens_html = "".join(spans)

            # --- Stage 2: Full BPE tokenization (corrected) ---
            encoded = tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)
            # The fast tokenizer returns plain Python lists (no tensors)
            input_ids = encoded["input_ids"]          # list of ints
            offsets = encoded["offset_mapping"]       # list of (start, end) tuples

            bpe_tokens = []
            for tid, (s, e) in zip(input_ids, offsets):
                subword_str = tokenizer.decode([tid])
                display_text = subword_str.replace(" ", "·")
                escaped_display = display_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                raw_bytes = subword_str.encode("utf-8")
                hex_bytes = " ".join(f"{b:02x}" for b in raw_bytes)
                bg, fg = hash_color(subword_str)

                bpe_tokens.append({
                    "id": tid,
                    "text": subword_str,
                    "display": escaped_display,
                    "start": s,
                    "end": e,
                    "bytes_hex": hex_bytes,
                    "byte_len": len(raw_bytes),
                    "bg": bg,
                    "fg": fg
                })

            # --- Stage 3: Decode ---
            decoded_text = tokenizer.decode(input_ids)

    return render_template_string(
        HTML_TEMPLATE,
        input_text=input_text,
        regex_tokens_html=regex_tokens_html,
        regex_info=regex_info,
        regex_pattern=regex_pattern,
        bpe_tokens=bpe_tokens,
        decoded_text=decoded_text
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)