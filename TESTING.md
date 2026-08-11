# Phase 2 — Manual Testing Checklist

Use `data/sample.pdf` (run `python scripts/create_sample_pdf.py` if needed).

Start the app: `streamlit run app.py`

Ensure `.env` contains `LLM_API_KEY=sk-...` before testing answers.

---

## Test 1 — Direct question

**Question:** `What was the company's revenue in 2024?`

**Expected:**
- Answer mentions **$127.4 million**
- Sources include **Page 7**
- Retrieved chunks contain the financial highlights text

**What you're learning:** Basic retrieval + grounded generation works end-to-end.

---

## Test 2 — Rephrased question

**Question:** `How much money did the firm make in twenty twenty-four?`

**Expected:**
- Same factual answer ($127.4 million) despite different wording
- Similarity scores may be lower than Test 1 — that's normal
- Page 7 should still appear in sources

**What you're learning:** Embeddings capture *meaning*, not exact keyword match.

---

## Test 3 — Multi-page question

**Question:** `What are the company's products and financial performance?`

**Expected:**
- Answer may combine products (page 2) and revenue (page 7)
- Sources may list **multiple pages**
- Multiple chunks retrieved with moderate similarities

**What you're learning:** `top_k > 1` lets the LLM see information from different parts of the document.

---

## Test 4 — Missing information

**Question:** `What is the population of Japan?`

**Expected:**
- Answer says information is **not available in the document**
- Low-confidence warning may appear (similarity below threshold)
- Retrieved chunks are unrelated (financial/contact pages)

**What you're learning:** RAG should refuse to invent facts. Threshold is a hint, not a guarantee — always read the retrieved chunks.

---

## Test 5 — Similar wording, different meaning

**Question:** `What was the revenue in 2023?`

**Expected:**
- Answer: **$104.3 million** (page 8), NOT 2024's $127.4M
- Watch whether retrieval picks page 8 over page 7

**What you're learning:** Frequent words like "revenue" appear on many pages — retrieval must distinguish the *specific* fact asked for.

---

## Test 6 — Cross-questioning / understanding

**Question 1:** `Who is the CEO?`  
**Question 2:** `When did that person join the company?`

**Expected:**
- Q1: Maria Chen, page 3
- Q2: Joined in **2019** (same leadership chunk if retrieved)

**Note:** There is no conversation memory in Phase 2 — each question is independent. Q2 must retrieve the CEO chunk on its own.

**What you're learning:** RAG retrieves per question; follow-ups don't automatically carry context (that would be Phase 3+).

---

## Debugging tips

| Symptom | Likely cause |
|---------|----------------|
| Wrong answer, good chunks | LLM ignored context — check prompt |
| Wrong answer, wrong chunks | Retrieval problem — tune chunk size / top_k |
| "Not available" but answer is in doc | Retrieval missed it — try rephrasing or increase top_k |
| Confident wrong answer | Hallucination — strengthen prompt or improve retrieval |
| Low similarity warning always | Threshold too high — lower it in sidebar |

---

## Phase 1 CLI (retrieval only)

Still available for inspecting chunks without LLM:

```bash
cd src
python main.py
```
