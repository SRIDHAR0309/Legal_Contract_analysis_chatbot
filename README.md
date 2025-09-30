## GenAI-powered Legal Contract Analysis and Risk Assessment Bot

### Overview
A modular Python + Streamlit app that lets users upload contracts (PDF/DOCX/TXT), preprocesses them with spaCy + regex, runs a mock LLM analysis (GPT-4/Claude stub) to identify clause-level risks, and outputs an overall risk score with export options (JSON and PDF).

### Tech Stack
- Backend: Python 3.9+, spaCy, PyPDF2, docx2txt, regex
- Frontend: Streamlit, Plotly
- Exports: JSON, PDF (reportlab)
- LLM: GPT-4/Claude (stubbed; easy to wire in real API)

### Quickstart
1. Create and activate a virtual environment (recommended)
2. Install dependencies
3. Run the app

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Notes
- If the spaCy `en_core_web_sm` model is not installed, the app falls back to a lightweight sentencizer so it can run offline within 2 minutes.
- Replace the mock LLM in `backend/analysis.py` with real GPT-4/Claude calls by providing your API key and provider.

### Project Structure
```
backend/
  __init__.py
  extraction.py
  preprocessing.py
  analysis.py
  scoring.py
  export.py
app.py
```

### License
MIT
"# Legal_Contract_analysis_chatbot" 
