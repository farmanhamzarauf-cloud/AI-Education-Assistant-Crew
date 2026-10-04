# AI Education Assistant (Learn Mate AI)

A single-agent CrewAI app that researches a topic with DuckDuckGo and writes a report using Groq (`openai/gpt-oss-120b`). UI built with Streamlit.

## Deploy on Streamlit Cloud
1. Upload these files to a GitHub repository.
2. On share.streamlit.io, create an app: choose the repo, branch `main`, main file `app.py`.
3. In Advanced settings, select Python 3.11 and add this under Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

4. Click Deploy.
