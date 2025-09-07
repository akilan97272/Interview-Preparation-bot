# interview_simulator.py
import streamlit as st
import os
import json
import base64
import google.generativeai as genai
import re

# ---- CONFIG ----
GEMINI_API_KEY="AIzaSyDH1H2P519Sql1bMu-gExULuhnPKao32Do" 
if not GEMINI_API_KEY:
    st.warning("Set GEMINI_API_KEY env var before running the app.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Choose a model (Flash is faster/cheaper, Pro is smarter)
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# ---- PROMPT TEMPLATES ----
SYSTEM_PROMPT = """
You are an expert technical interviewer and coach. Ask role- and domain-appropriate questions. 
For each candidate answer, evaluate clarity, correctness (technical) or examples (behavioral), completeness, and structure. 
Return JSON when requested.
"""

QUESTION_GEN_TEMPLATE = """Generate {n_questions} interview questions for:
role: {role}
domain: {domain}
mode: {mode}
difficulty: {difficulty}

Return JSON list: [{{"id":1,"question":"...","type":"coding|system-design|behavioral","expected_topics":["..."]}}, ...]
"""

EVAL_PROMPT_TEMPLATE = """You are an interviewer-evaluator.
Question: {question}
Candidate answer: {answer}
Mode: {mode}
Role: {role}

Score and explain using this rubric:
- clarity (0-2)
- accuracy_or_examples (0-3)
- completeness (0-2)
- structure (0-2)

Return strictly JSON with fields: scores, total, feedback (list), suggested_resources (list).
"""


def safe_json_loads(text: str):
    """
    Try to parse JSON robustly:
    1) try direct json.loads
    2) try to extract a JSON substring ([...] or {...})
    3) try replacing single quotes with double quotes and parse
    Returns Python object on success, else None.
    """
    if not isinstance(text, str):
        return None
    cleaned = clean_response(text)

    # 1) direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2) try to extract JSON-like substring
    m = re.search(r'(\[.*\]|\{.*\})', cleaned, re.S)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) try simple fix: single-quotes -> double-quotes
    try:
        alt = cleaned.replace("'", '"')
        return json.loads(alt)
    except Exception:
        pass

    return None



def clean_response(text: str) -> str:
    """Remove code fences and trim whitespace."""
    text = re.sub(r"```(?:json|python)?", "", text)
    text = text.replace("```", "")
    return text.strip()

def generate_questions(role, domain, mode, difficulty, n_questions):
    p = f"Generate {n_questions} interview questions for role {role} in domain {domain} with mode {mode} and difficulty {difficulty}."
    out = call_llm(p)                   
    cleaned_out = clean_response(out)

    try:
        questions = json.loads(cleaned_out)  # if JSON
        # normalize JSON into [{"question": ...}, ...]
        if isinstance(questions, list):
            return [{"question": q.get("question", q) if isinstance(q, dict) else str(q)} for q in questions]
        elif isinstance(questions, dict) and "questions" in questions:
            return [{"question": q} for q in questions["questions"]]
        else:
            return [{"question": str(questions)}]
    except json.JSONDecodeError:
        # fallback: extract lines that look like questions
        lines = []
        for line in cleaned_out.splitlines():
            s = line.strip()
            if not s:
                continue
            # Only keep lines that look like actual questions
            if re.match(r'^\d+[\.\)]', s) or s.lower().startswith("q"):
                s = re.sub(r'^\s*(?:Q\s*)?\d+\s*[\.\):-]\s*', '', s)  # strip numbering like "Q1.", "1)", "1."
                lines.append(s)
        # If nothing matched, just take sentences ending with '?'
        if not lines:
            lines = [sent.strip() for sent in re.split(r'(?<=\?)', cleaned_out) if sent.strip().endswith('?')]
        return [{"question": q} for q in lines][:n_questions]



def extract_questions(text: str):
    # Remove markdown fences first
    text = re.sub(r"```(?:json|python)?", "", text).strip()
    try:
        data = json.loads(text)
        return data.get("questions", data)  # depends on how your prompt is structured
    except json.JSONDecodeError:
        return [text]  # fallback: just return raw


# ---- HELPERS ----
def call_llm(prompt: str, system=SYSTEM_PROMPT, max_tokens=800):
    # Combine system + user text (Gemini doesn’t support "system" role)
    full_prompt = f"{system.strip()}\n\nUser request:\n{prompt.strip()}"

    resp = model.generate_content(
        [full_prompt],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
    )
    return resp.text

def evaluate_answer(question, answer, mode, role):
    p = EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer, mode=mode, role=role)
    out = call_llm(p, max_tokens=500)
    try:
        jobj = json.loads(out)
        return jobj
    except Exception:
        # fallback: extract JSON substring if wrapped in text
        import re
        m = re.search(r'(\{.*\})', out, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
        # safe fallback
        return {
            "scores": {"clarity": 0, "accuracy_or_examples": 0, "completeness": 0, "structure": 0},
            "total": 0,
            "verdict": "Could not evaluate",
            "feedback": ["LLM response was not in JSON format."],
            "suggested_resources": []
        }


def download_link(text: str, filename: str):
    b = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b}" download="{filename}">Download {filename}</a>'
    return href

# ---- UI ----
st.set_page_config(page_title="Interview Simulator", layout="wide")
st.title("Interview Simulator")

with st.sidebar:
    st.header("Session settings")
    role = st.selectbox("Role", ["Software Engineer", "Data Analyst", "Product Manager", "Frontend Engineer", "Backend Engineer"])
    domain = st.text_input("Domain (optional)", value="backend")
    mode = st.selectbox("Mode", ["Technical", "Behavioral"])
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    n_questions = st.slider("Number of questions", 3, 6, 4)
    if st.button("Generate Questions"):
        st.session_state['questions'] = generate_questions(role, domain, mode, difficulty, n_questions)
        st.session_state['current_q'] = 0
        st.session_state['answers'] = []
        st.session_state['evaluations'] = []

if 'questions' not in st.session_state:
    st.session_state['questions'] = []
    st.session_state['current_q'] = 0
    st.session_state['answers'] = []
    st.session_state['evaluations'] = []

cols = st.columns([3,2])
with cols[0]:
    st.subheader("Interview")
    if not st.session_state['questions']:
        st.info("Generate questions from the sidebar to start.")
    else:
        idx = st.session_state['current_q']
        qobj = st.session_state['questions'][idx]
        st.markdown(f"**Q{idx+1}. {qobj.get('question')}**")
        ans = st.text_area("Your answer", key=f"answer_{idx}", height=200)
        c1, c2, c3 = st.columns(3)
        if c1.button("Submit Answer", key=f"submit_{idx}"):
            if not ans.strip():
                st.warning("Please write an answer or skip.")
            else:
                st.session_state['answers'].append({"q":qobj.get('question'), "a": ans})
                with st.spinner("Evaluating..."):
                    ev = evaluate_answer(qobj.get('question'), ans, mode, role)
                    st.session_state['evaluations'].append(ev)
                st.success("Evaluated. See feedback panel.")
        if c2.button("Skip Question", key=f"skip_{idx}"):
            st.session_state['answers'].append({"q":qobj.get('question'), "a": None})
            st.session_state['evaluations'].append({"skipped": True})
            st.success("Skipped.")
        if c3.button("Next", key=f"next_{idx}"):
            if idx < len(st.session_state['questions']) - 1:
                st.session_state['current_q'] += 1
            else:
                st.info("You are at the last question.")

with cols[1]:
    st.subheader("Feedback & Scores")
    if st.session_state['evaluations']:
        st.json(st.session_state['evaluations'][-1])
    else:
        st.write("No feedback yet. Submit an answer to evaluate.")

st.write("---")
if st.button("Finish & Generate Summary"):
    total_scores, feedbacks = [], []
    for ev in st.session_state['evaluations']:
        if ev.get('total') is not None:
            total_scores.append(ev['total'])
        if ev.get('feedback'):
            feedbacks.extend(ev['feedback'])
    avg = sum(total_scores)/len(total_scores) if total_scores else 0
    summary = {
        "role": role,
        "domain": domain,
        "mode": mode,
        "average_score": avg,
        "feedbacks": feedbacks,
        "evaluations": st.session_state['evaluations'],
        "answers": st.session_state['answers'],
        "questions": st.session_state['questions']
    }
    st.header("Session Summary")
    st.write("Average score:", avg)
    for f in feedbacks[:6]:
        st.write("- ", f)
    txt = json.dumps(summary, indent=2)
    st.markdown(download_link(txt, "session_summary.json"), unsafe_allow_html=True)
