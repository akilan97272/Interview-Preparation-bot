# interview_simulator.py
import streamlit as st
import os
import json
from typing import List, Dict
import time
import base64

# OpenAI SDK
import openai

# ---- CONFIG ----
OPENAI_API_KEY = os.getenv("gpt-5-nano-2025-08-07")
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY env var before running the app.")
openai.api_key = OPENAI_API_KEY

MODEL = "gpt-4o"  # change to available model for your account

# ---- PROMPT TEMPLATES ----
SYSTEM_PROMPT = """
You are an expert technical interviewer and coach. Ask role- and domain-appropriate questions. For each candidate answer, evaluate clarity, correctness (technical) or examples (behavioral), completeness, and structure. Return JSON when requested.
"""

QUESTION_GEN_TEMPLATE = """Generate {n_questions} interview questions for:
role: {role}
domain: {domain}
mode: {mode}
difficulty: {difficulty}

Return JSON list: [{"id":1,"question":"...","type":"coding|system-design|behavioral","expected_topics":["..."]}, ...]
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

# ---- HELPERS ----
def call_llm(prompt: str, system=SYSTEM_PROMPT, max_tokens=800):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2
    )
    return resp['choices'][0]['message']['content']

def generate_questions(role, domain, mode, difficulty, n_questions=4):
    p = QUESTION_GEN_TEMPLATE.format(n_questions=n_questions, role=role, domain=domain, mode=mode, difficulty=difficulty)
    out = call_llm(p)
    # try parse JSON; if it fails, return fallback
    try:
        qlist = json.loads(out)
        return qlist
    except Exception:
        # fallback: wrap lines
        return [{"id": i+1, "question": q.strip(), "type": "general", "expected_topics": []}
                for i,q in enumerate(out.splitlines()) if q.strip()][:n_questions]

def evaluate_answer(question, answer, mode, role):
    p = EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer, mode=mode, role=role)
    out = call_llm(p, max_tokens=500)
    try:
        jobj = json.loads(out)
        return jobj
    except Exception:
        # if LLM returns text, attempt to extract JSON substring
        import re
        m = re.search(r'(\{.*\})', out, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
        # fallback minimal
        return {"scores":{"clarity":1,"accuracy_or_examples":1,"completeness":1,"structure":1},
                "total":4,"feedback":["Could not parse automatic evaluation reliably."],"suggested_resources":[]}

def download_link(text: str, filename: str):
    b = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b}" download="{filename}">Download {filename}</a>'
    return href

# ---- UI ----
st.set_page_config(page_title="Interview Simulator", layout="wide")
st.title("LLM Interview Simulator — MVP")

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
        last = st.session_state['evaluations'][-1]
        st.json(last)
    else:
        st.write("No feedback yet. Submit an answer to evaluate.")

st.write("---")
if st.button("Finish & Generate Summary"):
    # produce summary
    total_scores = []
    feedbacks = []
    for ev in st.session_state['evaluations']:
        if ev.get('total') is not None:
            total_scores.append(ev.get('total'))
        if ev.get('feedback'):
            feedbacks.extend(ev.get('feedback'))
    average = sum(total_scores)/len(total_scores) if total_scores else 0
    summary = {
        "role": role,
        "domain": domain,
        "mode": mode,
        "n_questions": len(st.session_state['questions']),
        "average_score": average,
        "feedbacks": feedbacks,
        "evaluations": st.session_state['evaluations'],
        "answers": st.session_state['answers'],
        "questions": st.session_state['questions']
    }
    st.header("Session Summary")
    st.write("Average score:", average)
    st.write("Top feedback items:")
    for f in feedbacks[:6]:
        st.write("- ", f)
    txt = json.dumps(summary, indent=2)
    st.markdown(download_link(txt, "session_summary.json"), unsafe_allow_html=True)

    # simple PDF export using reportlab (optional)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        fn = "/tmp/session_summary.pdf"
        c = canvas.Canvas(fn, pagesize=letter)
        text = c.beginText(40, 750)
        text.setFont("Helvetica", 10)
        lines = [
            f"Role: {role}",
            f"Domain: {domain}",
            f"Mode: {mode}",
            f"Average score: {average}",
            "",
            "Feedback summary:"
        ]
        lines += feedbacks[:40]
        for L in lines:
            text.textLine(L)
        c.drawText(text)
        c.save()
        with open(fn, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="session_summary.pdf">Download PDF summary</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.info("PDF export not available (reportlab not installed).")

st.write("Tip: For more reliable automated scoring, tune prompts & use a specialized rubric per question type.")
