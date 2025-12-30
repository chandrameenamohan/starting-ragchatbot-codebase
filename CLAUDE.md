# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials using semantic search and Claude AI. The system processes course documents, stores them as vector embeddings in ChromaDB, and uses tool-based agentic responses to search and synthesize answers.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Architecture

### Request Flow

```
Frontend (script.js)
    → POST /api/query
    → app.py (FastAPI endpoint)
    → rag_system.py (orchestrator)
    → ai_generator.py (Claude API with tools)
    → Claude decides to use search_course_content tool
    → search_tools.py (tool execution)
    → vector_store.py (ChromaDB semantic search)
    → Results formatted and returned to Claude
    → Claude synthesizes final answer
    → Response with sources returned to frontend
```

### Backend Components (`backend/`)

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server, API endpoints, startup document loading |
| `rag_system.py` | Main orchestrator - coordinates all components |
| `ai_generator.py` | Claude API integration with tool-calling loop |
| `vector_store.py` | ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `document_processor.py` | Parses course documents, chunks text with sentence-aware overlap |
| `search_tools.py` | Tool definitions and execution for Claude's `search_course_content` tool |
| `session_manager.py` | In-memory conversation history per session |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |
| `config.py` | Configuration dataclass with defaults |

### Frontend (`frontend/`)

Vanilla HTML/CSS/JS chat interface. Uses `marked.js` for markdown rendering. Communicates via `/api/query` and `/api/courses` endpoints.

### Document Format (`docs/`)

Course documents must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [Lesson Title]
Lesson Link: [url]
[content...]

Lesson 1: [Lesson Title]
[content...]
```

### Key Configuration (`config.py`)

- `CHUNK_SIZE`: 800 chars per chunk
- `CHUNK_OVERLAP`: 100 chars overlap between chunks
- `MAX_RESULTS`: 5 search results returned
- `MAX_HISTORY`: 2 conversation exchanges remembered
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

### ChromaDB Collections

1. **course_catalog**: Stores course metadata (title, instructor, links, lessons JSON) for semantic course name resolution
2. **course_content**: Stores chunked course content with metadata (course_title, lesson_number, chunk_index)

## Environment Setup

Requires `.env` file with:
```
ANTHROPIC_API_KEY=your_key_here
```

## Development Guidelines

- Always use `uv` to manage packages, do not use `pip` directly
- Run `./dev.sh format` before committing to ensure consistent code style

## Code Quality

The project uses `black` for code formatting. Run quality checks with:

```bash
# Format all Python files
./dev.sh format

# Check formatting without changes
./dev.sh check

# Run tests
./dev.sh test

# Run all quality checks (format check + tests)
./dev.sh all
```

Black configuration is in `pyproject.toml` with line-length of 88 characters.
