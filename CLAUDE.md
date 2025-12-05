# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Han-Tyumi is a Streamlit-based chatbot that answers questions about King Gizzard & The Lizard Wizard. It uses LangChain to route queries to either:
- A SQLite database for setlist/discography data
- A DeepLake vector store for interview/contextual data

The bot responds in character as Han-Tyumi, a cyborg character from the band's lore.

## Commands

```bash
# Install dependencies
poetry install

# Run locally
poetry run streamlit run han_tyumi/main.py

# Deploy to Fly.io
fly deploy
```

## Architecture

**Query Routing** (`han_tyumi/main.py`): A LangChain classifier routes questions to:
- `SetListData` / `Lyrics`: SQL chain queries `kglw.db` (SQLite)
- `InterviewData` / `Other`: RAG chain queries DeepLake vector store

**Database**: SQLite with tables: `setlists`, `shows`, `albums`, `songs`, `tracks`, `tours`, `venues`

**Required Secrets** (via Streamlit secrets):
- `OPENAI_API_KEY`
- `ACTIVELOOP_TOKEN`
- `DATASET_PATH` (DeepLake path)
- `DB_FILE_URI` (remote SQLite download URL)

## SQLite Query Notes

When writing SQL for this database:
- Use `JulianDay()` for date arithmetic (no DATEDIFF)
- Use `strftime('%w', date)` for day of week (no DAYOFWEEK)
- No need to filter by band name - all data is for King Gizzard
