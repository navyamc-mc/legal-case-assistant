# Legal Case Q&A (Hackathon Prototype)

Pick a case from your uploaded documents and ask questions. Answers come only from that case's
judgments, orders and proofs, with the source file and page shown.

## Project structure

```
legal-case-qa/
  app.py            # frontend + backend (Streamlit app)
  ingest.py         # reads documents and builds the search database
  requirements.txt
  cases/            # YOUR DATA: one sub-folder per case
    Sample_Ravi_vs_Kumar_2022/
      judgment.txt
  chroma_db/        # created automatically by ingest.py
```

## How it works

1. `ingest.py` reads every PDF/TXT in each case folder, splits it into chunks, and stores them in
   ChromaDB with metadata (case name, file, page, document type).
2. `app.py` shows a case search box. When you ask a question, it finds the 6 most relevant chunks
   from the selected case only, sends them to Claude, and displays the answer with sources.

## Setup and run

You need Python 3.10 or newer.

**1. Open a terminal in the project folder and create a virtual environment**

```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

**2. Install the packages**

```bash
pip install -r requirements.txt
```

**3. Get a FREE AI key (no payment needed)**

Go to **aistudio.google.com**, sign in with a Google account, click **Get API key**, then **Create API key**. Then set it in your terminal:

```bash
# Windows Command Prompt
set GEMINI_API_KEY=your-key-here
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
# Mac / Linux
export GEMINI_API_KEY="your-key-here"
```

Free-tier limits apply and Google may use free-tier inputs to improve its models, so use sample or public
cases only, not confidential client files. If the default model name stops working, set another one,
for example `GEMINI_MODEL=gemini-3-flash-preview`.

Other options: `LLM_PROVIDER=ollama` runs a free model on your own computer with no key (install Ollama, then
`ollama pull llama3.2`), and `LLM_PROVIDER=anthropic` uses Claude (paid, needs `ANTHROPIC_API_KEY` and `pip install anthropic`).

**4. Add your cases**

Inside `cases/`, make one folder per case and put its PDFs in it. The folder name becomes the case name.
Use words like `judgment`, `order`, `proof` or `exhibit` in file names so the document type is tagged.

```
cases/
  State_vs_Sharma_2021/
    judgment.pdf
    order_2021-03-04.pdf
    proof_exhibit_A.pdf
```

**5. Build the search database** (run again whenever you add or change documents)

```bash
python ingest.py
```

**6. Start the app**

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

The first run of `ingest.py` downloads a small embedding model (about 80 MB), so it needs internet once.

## Demo script (2 minutes)

1. Search a case name in the sidebar and select it.
2. Click **Generate summary**.
3. Ask: "What was the final order?", "Who were the witnesses?", "What evidence was produced?"
4. Ask something not in the documents to show it says "not found" instead of guessing.
5. Open **Sources used** to show citations.

## Known limits (mention these honestly in your pitch)

- **Scanned PDFs** (images of pages) have no text, so they need OCR first. `ingest.py` warns you
  about them. Quick fix: run them through `ocrmypdf` before ingesting.
- The summary uses the first 30 chunks of the case, so for very long cases it can miss later parts.
- English works best. For Hindi or Kannada documents, swap in a multilingual embedding model.
- This is a research aid, not legal advice. Always verify against the original documents.

## Ideas to extend if you have time

- Timeline of orders and hearing dates
- Compare two cases side by side
- Upload new documents from inside the app
- Login so different users see different cases


## Methods used (for your viva)

| Method | Where in the project |
|---|---|
| **RAG** | `app.py`: retrieve chunks, then answer only from them |
| **Vector database** | ChromaDB in `ingest.py`, with metadata filter per case |
| **Chunking** | `ingest.py`: section-aware (FACTS, ISSUES, REASONING, FINAL ORDER) or fixed 300-word chunks with 60-word overlap. Switch with `CHUNK_STRATEGY=words` |
| **Harness engineering** | `app.py`: system-prompt rules, case-filter guardrail, "not found" refusal, error handling, provider switch, and `verify_citations()` which checks every citation against the retrieved text |
| **Benchmark** | `evaluate.py` + `questions.json`: Hit@3 and MRR for both chunking strategies. Run `python evaluate.py` |

Note: the sample documents are short, so both strategies may score similarly. Test with long real PDFs to see differences.
