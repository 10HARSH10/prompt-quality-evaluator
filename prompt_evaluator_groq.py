import html
import json
import re

import streamlit as st
from groq import Groq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Quality Evaluator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
  .stApp { background: #0e0e0e; color: #e8e8e0; }

  .metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #666;
    margin-bottom: 4px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-score {
    font-size: 2.2rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-feedback { font-size: 13px; color: #aaa; margin-top: 6px; line-height: 1.5; }

  .score-bar-bg {
    background: #2a2a2a;
    border-radius: 4px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
  }
  .score-bar-fill { height: 6px; border-radius: 4px; }

  .overall-score {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .overall-number {
    font-size: 4rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
  }
  .overall-label { font-size: 12px; color: #555; text-transform: uppercase; letter-spacing: 0.12em; }

  div[data-testid="stTextArea"] textarea {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e8e8e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
  }
  div[data-testid="stButton"] > button {
    background: #e8e8e0 !important;
    color: #0e0e0e !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    padding: 0.5rem 2rem !important;
    font-size: 13px !important;
  }
  div[data-testid="stButton"] > button:hover { background: #c8c8c0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Groq client setup (free tier API key from console.groq.com) ───────────────
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("⚠️ Groq API key not found.")
    st.info(
        "Get a free key at console.groq.com, then in Streamlit Cloud go to "
        "Settings → Secrets and paste:\n\nGROQ_API_KEY = \"your-key-here\""
    )
    st.stop()

client = Groq(api_key=api_key)

# ── Models ────────────────────────────────────────────────────────────────────
# Tried in order. If one is retired or unavailable, the app falls back to the
# next, so a future model shutdown won't break it. Check console.groq.com/docs/models
# for the current list and edit here if needed.
MODEL_CHAIN = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]

# ── Constants ─────────────────────────────────────────────────────────────────
PARAMETERS = ["Relevance", "Clarity", "Completeness", "Tone", "Accuracy"]

SYSTEM_PROMPT = """You are an expert LLM prompt-response quality evaluator.
Given a PROMPT and a RESPONSE, score the response across 5 parameters on a scale of 1-10.
Return ONLY valid JSON with absolutely no markdown fences, no explanation, no extra text.
Use this exact structure:
{
  "Relevance":    {"score": <int 1-10>, "feedback": "<one specific sentence>"},
  "Clarity":      {"score": <int 1-10>, "feedback": "<one specific sentence>"},
  "Completeness": {"score": <int 1-10>, "feedback": "<one specific sentence>"},
  "Tone":         {"score": <int 1-10>, "feedback": "<one specific sentence>"},
  "Accuracy":     {"score": <int 1-10>, "feedback": "<one specific sentence>"},
  "overall_feedback": "<2-3 sentence summary with one actionable improvement>"
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_color(score):
    if score >= 8:
        return "#4caf50"
    if score >= 5:
        return "#ffc107"
    return "#f44336"


def overall_score(result):
    return round(sum(result[p]["score"] for p in PARAMETERS) / len(PARAMETERS), 1)


def extract_json(raw):
    """Pull the JSON object out of the model output, even if it added fences or extra text."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    return json.loads(raw[start:end + 1])


def normalize(result):
    """Validate the model's JSON and clamp scores to 1-10 so the UI never breaks."""
    clean = {}
    for p in PARAMETERS:
        item = result[p]
        score = int(round(float(item["score"])))
        clean[p] = {
            "score": max(1, min(10, score)),
            "feedback": str(item.get("feedback", "")),
        }
    clean["overall_feedback"] = str(result.get("overall_feedback", ""))
    return clean


def call_model(model, user_content, use_json_mode):
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content


def evaluate_pair(prompt_text, response_text):
    user_content = f"PROMPT:\n{prompt_text}\n\nRESPONSE:\n{response_text}"
    last_error = None

    for model in MODEL_CHAIN:
        for use_json_mode in (True, False):
            try:
                raw = call_model(model, user_content, use_json_mode)
                result = normalize(extract_json(raw))
                st.session_state["active_model"] = model
                return result
            except json.JSONDecodeError as e:
                last_error = e
                continue
            except (KeyError, ValueError, TypeError) as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                # Model missing/retired: move on to the next model in the chain.
                if any(k in msg for k in ("not exist", "decommission", "not found", "404", "no access")):
                    break
                # JSON mode unsupported: retry same model without it.
                if use_json_mode and any(k in msg for k in ("response_format", "json")):
                    continue
                raise

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def render_result(result, label=""):
    avg = overall_score(result)
    color = score_color(avg)

    st.markdown(f"""
    <div class="overall-score">
      <div class="overall-label">{label + " " if label else ""}Overall Score</div>
      <div class="overall-number" style="color:{color}">{avg}<span style="font-size:1.5rem;color:#444">/10</span></div>
    </div>""", unsafe_allow_html=True)

    for param in PARAMETERS:
        s = result[param]["score"]
        fb = html.escape(result[param]["feedback"])
        c = score_color(s)
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{param}</div>
          <div class="metric-score" style="color:{c}">{s}<span style="font-size:1rem;color:#444">/10</span></div>
          <div class="score-bar-bg"><div class="score-bar-fill" style="width:{s*10}%;background:{c}"></div></div>
          <div class="metric-feedback">{fb}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="border-color:#333">
      <div class="metric-label">Overall Feedback</div>
      <div class="metric-feedback" style="color:#ccc;font-size:14px">{html.escape(result['overall_feedback'])}</div>
    </div>""", unsafe_allow_html=True)


def show_error(e):
    msg = str(e)
    if "429" in msg or "rate" in msg.lower():
        st.error("Rate limit reached on the free tier. Wait a minute and try again.")
    else:
        st.error(f"Error: {msg}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎯 Prompt Quality Evaluator")
st.markdown(
    '<p style="color:#666;font-family:\'IBM Plex Mono\',monospace;font-size:13px;">'
    f'Powered by Groq (free tier) · {MODEL_CHAIN[0]} · Score · Evaluate · Compare · Improve</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡ Single Evaluation", "⚖️ Compare Responses", "📚 Learn the Parameters"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Evaluation
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Evaluate a prompt-response pair")
    st.caption("Paste any prompt and an AI-generated response. It will be scored across 5 quality parameters.")

    col1, col2 = st.columns(2)
    with col1:
        prompt_input = st.text_area(
            "✍️ Prompt", height=160,
            placeholder="Enter the prompt that was given to the AI...", key="single_prompt")
    with col2:
        response_input = st.text_area(
            "🤖 AI Response", height=160,
            placeholder="Enter the AI-generated response to evaluate...", key="single_response")

    if st.button("Evaluate →", key="btn_single"):
        if not prompt_input.strip() or not response_input.strip():
            st.warning("Please enter both a prompt and a response.")
        else:
            with st.spinner("Evaluating across 5 parameters..."):
                try:
                    result = evaluate_pair(prompt_input, response_input)
                    st.markdown("---")
                    st.markdown("### Results")
                    render_result(result)
                    st.caption(f"Scored by {st.session_state.get('active_model', MODEL_CHAIN[0])}")
                except Exception as e:
                    show_error(e)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Compare Responses
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Compare two responses to the same prompt")
    st.caption("Submit the same prompt with two different AI responses. Scores both and declares a winner.")

    compare_prompt = st.text_area(
        "✍️ Shared Prompt", height=100,
        placeholder="The prompt both responses are answering...", key="compare_prompt")

    col_a, col_b = st.columns(2)
    with col_a:
        response_a = st.text_area(
            "Response A", height=200, placeholder="First AI response...", key="resp_a")
    with col_b:
        response_b = st.text_area(
            "Response B", height=200, placeholder="Second AI response...", key="resp_b")

    if st.button("Compare Both →", key="btn_compare"):
        if not compare_prompt.strip() or not response_a.strip() or not response_b.strip():
            st.warning("Please fill in the prompt and both responses.")
        else:
            with st.spinner("Scoring both responses..."):
                try:
                    result_a = evaluate_pair(compare_prompt, response_a)
                    result_b = evaluate_pair(compare_prompt, response_b)

                    avg_a = overall_score(result_a)
                    avg_b = overall_score(result_b)

                    st.markdown("---")
                    st.markdown("### Comparison Results")

                    if avg_a == avg_b:
                        banner = f"🤝 It's a tie — {avg_a}/10 each"
                    else:
                        winner = "A" if avg_a > avg_b else "B"
                        banner = f"🏆 Response {winner} wins — {max(avg_a, avg_b)}/10 vs {min(avg_a, avg_b)}/10"

                    st.markdown(f"""
                    <div style="background:#0a2a0a;border:1px solid #1a4a1a;border-radius:8px;
                         padding:1rem 1.5rem;margin-bottom:1.5rem;text-align:center">
                      <span style="color:#4caf50;font-family:'IBM Plex Mono',monospace;
                            font-size:1.1rem;font-weight:600">{banner}</span>
                    </div>""", unsafe_allow_html=True)

                    col_ra, col_rb = st.columns(2)
                    with col_ra:
                        st.markdown("**Response A**")
                        render_result(result_a, "A")
                    with col_rb:
                        st.markdown("**Response B**")
                        render_result(result_b, "B")

                    st.markdown("### Parameter-by-Parameter Breakdown")
                    rows = ["| Parameter | Response A | Response B | Winner |",
                            "|-----------|:----------:|:----------:|:------:|"]
                    for p in PARAMETERS:
                        sa, sb = result_a[p]["score"], result_b[p]["score"]
                        w = "A ✅" if sa > sb else ("B ✅" if sb > sa else "Tie")
                        rows.append(f"| {p} | {sa}/10 | {sb}/10 | {w} |")
                    st.markdown("\n".join(rows))

                except Exception as e:
                    show_error(e)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Learn the Parameters
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### The 5 Evaluation Parameters")
    st.caption("Everything you need to explain these parameters confidently in your interview.")

    params_info = {
        "Relevance": {
            "icon": "🎯",
            "definition": "Does the response directly answer what the prompt asked? A highly relevant response stays on-topic and doesn't introduce unnecessary information.",
            "high": "The response addresses the exact question or task without going off-topic.",
            "low": "The response talks around the topic, answers a different question, or pads with unrelated content.",
            "tip": '"I check if every sentence serves the prompt. If it can be removed without losing the answer, it hurts relevance."',
        },
        "Clarity": {
            "icon": "💡",
            "definition": "Is the response easy to read and understand? Good clarity means clear sentence structure, no unexplained jargon, and logical flow.",
            "high": "A first-time reader understands it without re-reading. Short sentences, active voice, well-structured.",
            "low": "Confusing structure, undefined jargon, contradictions, or a wall of text with no formatting.",
            "tip": '"I ask: would a 16-year-old understand this on the first read? If not, clarity needs work — even in technical content."',
        },
        "Completeness": {
            "icon": "✅",
            "definition": "Does the response cover all parts of the prompt? A complete response doesn't leave out important sub-questions or aspects.",
            "high": "All parts of the prompt are addressed. If the prompt had 3 questions, all 3 are answered.",
            "low": "Answers only part of what was asked, leaves key details out, or stops mid-explanation.",
            "tip": '"I map each clause in the prompt to a section of the response. Anything unmapped is a completeness gap."',
        },
        "Tone": {
            "icon": "🎭",
            "definition": "Is the tone appropriate for the context? Customer service = warm; technical spec = precise; creative story = engaging.",
            "high": "Tone matches the intended audience and purpose — formal where needed, conversational where appropriate.",
            "low": "Too casual for a professional context, too robotic for a creative task, or inconsistent throughout.",
            "tip": '"Tone is like dress code for language. I check if the response is wearing the right outfit for the occasion."',
        },
        "Accuracy": {
            "icon": "🔬",
            "definition": "Are the facts, logic, and claims in the response correct? Critical for educational or informational prompts.",
            "high": "All stated facts are verifiable and correct. Reasoning is logically sound with no contradictions.",
            "low": "Contains hallucinated facts, logical errors, incorrect data, or contradicts established knowledge.",
            "tip": '"Accuracy is hardest to score — it requires domain knowledge. When unsure, I flag for expert review rather than guessing."',
        },
    }

    for param, info in params_info.items():
        with st.expander(f"{info['icon']} {param}", expanded=False):
            st.markdown(f"**Definition:** {info['definition']}")
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**High score (8–10):** {info['high']}")
            with c2:
                st.error(f"**Low score (1–4):** {info['low']}")
            st.info(f"💬 **How to say it in your interview:**\n\n_{info['tip']}_")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#444;font-size:11px;font-family:\'IBM Plex Mono\',monospace">'
    'Prompt Quality Evaluator · Powered by Groq free tier · Built with Streamlit'
    '</p>',
    unsafe_allow_html=True,
)
