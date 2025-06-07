import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load Summarization and Embedding models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Extract text chunks from PDF
def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.split()) > 30]
            for para in paragraphs:
                chunks.append({'page': i + 1, 'text': para})
    return chunks

# Step 2: Summarize each chunk
def summarize_text(text):
    if len(text.split()) < 30:
        return text
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Step 3: Load maturity levels from DOCX file
def load_maturity_levels(docx_path):
    doc = Document(docx_path)
    levels = {}
    current_q = None
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if text.startswith("What") or text.startswith("To what") or text.startswith("How"):
            current_q = text
            levels[current_q] = []
        elif current_q and len(levels[current_q]) < 4:
            levels[current_q].append(text)
    return levels

# Step 4: Compare summary with maturity levels
def match_summary_to_level(summary, levels):
    embeddings = embedder.encode([summary] + levels, convert_to_tensor=True)
    sims = util.cos_sim(embeddings[0], embeddings[1:])[0]
    best_level = int(sims.argmax()) + 1
    return best_level, sims.tolist()

# Step 5: Run the full pipeline
def process(pdf_path, docx_path, questions):
    chunks = extract_chunks_from_pdf(pdf_path)
    levels_dict = load_maturity_levels(docx_path)
    results = []

    for question in questions:
        level_statements = levels_dict.get(question, [])
        if not level_statements:
            continue
        matched_chunks = [c for c in chunks if any(w.lower() in c['text'].lower() for w in question.split())]
        for chunk in matched_chunks[:3]:  # top 3 chunks per question
            summary = summarize_text(chunk['text'])
            level, sims = match_summary_to_level(summary, level_statements)
            results.append({
                'question': question,
                'page': chunk['page'],
                'original_chunk': chunk['text'],
                'summary': summary,
                'predicted_maturity_level': level,
                'similarities': sims
            })
    return pd.DataFrame(results)

# Step 6: Questions to evaluate
questions_list = [
    "What is the company’s purpose/vision/mission?",
    "To what extent use and how does the company manage external contractors / non-permanent /temporary employees?",
    "How does the company interact with its different stakeholders (customers, users, suppliers, employees, regulators, investors, government, society)?",
    "Is the company leveraging different disruptive and emerging technologies",
    "What are the key Metrics / KPIs / key performance indicators being tracked related to the innovation portfolio and overall business performance?",
    "Does the organization tolerate failure and encourage risk-taking?"
]

# Step 7: Run everything
df_result = process(
    pdf_path="RIL-Integrated-Annual-Report-2022-23.pdf",
    docx_path="Document.docx",
    questions=questions_list
)

# Step 8: Save output
df_result.to_excel("RIL_Maturity_Levels_LocalModel.xlsx", index=False)
print("✅ Results saved to RIL_Maturity_Levels_LocalModel.xlsx")
