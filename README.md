# 📚 Handbook Copilot

### RAG-Based AI Assistant for Institutional Documents

An AI-powered web application that lets users ask natural language questions about a 100+ page PDF and get cited, accurate answers with page numbers.

---

## 🎯 Problem Statement

Students, faculty, and employees waste hours searching through massive, dense PDF documents (like university handbooks, compliance policies, or grant guidelines) to find simple answers. Traditional `Ctrl+F` only matches exact words, so users miss information when their question is phrased differently from the document.

**Example:** A student asks *"What is the minimum attendance required?"* — but the handbook says *"75% course-wise attendance requirement"*. Ctrl+F fails. The student gives up.

---

## 💡 Solution

**Handbook Copilot** is a Retrieval-Augmented Generation (RAG) system that:
- Lets users ask questions in **plain English**
- Retrieves the most relevant sections from the PDF using **semantic search**
- Generates **cited answers** with exact page numbers
- Prevents hallucination by grounding responses in the document

---

## ✨ Features

- 💬 Natural language question answering
- 📖 Cited answers with page numbers
- 🎯 Semantic search (finds answers by meaning, not keywords)
- 🧠 Powered by Google Gemini LLM
- 🚫 No hallucination — AI only answers from the PDF
- 🖥️ Interactive chat interface built with Streamlit
- 🔒 Data stays local (no PDF uploaded to cloud)

---

## 🛠️ Technologies Used

### Language
- **Python 3.10+** — Core programming language

### Frontend
- **Streamlit** — Interactive chat UI (auto-generated from Python)

### Backend & AI
- **Google Gemini API** — Large Language Model (LLM) for generating answers
- **ChromaDB** — Vector database for storing and searching embeddings
- **LangChain Text Splitters** — Split PDF text into semantic chunks
- **pypdf** — Extract text from PDF documents
- **python-dotenv** — Securely manage API keys

### NLP & AI Techniques
- **Text Embeddings** — Convert text chunks into numerical vectors
- **Semantic Search** — Retrieve chunks by meaning, not keywords
- **Retrieval-Augmented Generation (RAG)** — Combine retrieval with LLM generation
- **Natural Language Generation** — Generate human-like answers via Gemini

---

## 📊 Dataset

This project demonstrates RAG on the **CMR University Student Handbook (Academic Year 2025-26)**.

| Attribute | Details |
| :--- | :--- |
| **Document** | CMR University Student Handbook |
| **Source** | CMR University official website |
| **Type** | PDF |
| **Size** | 33+ pages |
| **Language** | English |
| **Topics Covered** | Academic regulations, attendance policy, examinations (CIE/SEE), grading system (SGPA/CGPA), hostel facilities, placement centre, library, LEAP programme, code of conduct, student grievance redressal |
| **Total Chunks** | 120 |


1. Download any 100+ page text-based PDF
2. Place it at `data/handbook.pdf`
3. Run `python ingest.py`

The system works with **any PDF document** — the CMR handbook is used here as a demonstration.

---

## 🏗️ Architecture
