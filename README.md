# blind_sight_using_qdrant
An end-to-end multimodal textbook Q&amp;A system that answers text, image, and voice-based queries using page-accurate retrieval. Qdrant powers fast vector search for textbook text and diagrams, enabling scalable, semantic, and metadata-filtered retrieval across multiple classes and chapters.


# Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed for educational use. Textbook content is processed at the page level, ensuring that all answers are traceable to specific pages and diagrams from the source material.

# The system supports:

Text-based questions

Image/diagram-based questions

Speech-to-text input

Text-to-speech output

Multiple classes and chapters

# System Architecture
User (Text or Voice)
        |
        v
Speech-to-Text (optional)
        |
        v
Query Embedding
        |
        v
Qdrant Vector Search
        |
        v
Relevant Pages and Images
        |
        v
LLM-based Answer Generation
        |
        v
Text and Speech Output

# Key Features

Page-level grounding (each page is treated as the primary unit of knowledge)

Diagram-aware retrieval and explanation

Semantic search using vector embeddings

Metadata-based filtering (class, subject, chapter, page)

Offline speech-to-text support

Modular and reproducible pipeline

Docker-based vector database deployment

# Technologies Used

Python

PyMuPDF (PDF text and image extraction)

Sentence Transformers (text embeddings)

CLIP (image embeddings)

Qdrant (vector database)

Docker and Docker Compose

Vosk (offline speech-to-text)

pyttsx3 (text-to-speech)

Prerequisites

Ensure the following are installed:

Python 3.10 or 3.11

Docker and Docker Compose

Git

A working microphone (for speech input)



# Setup Instructions (End-to-End)
1. Clone the Repository
git clone https://github.com/Gurharshsingh/blind_sight_using_qdrant/tree/main
2. Create and Activate a Virtual Environment
python -m venv venv


Activate the environment:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

3. Install Python Dependencies
pip install -r requirements.txt

4. Start Qdrant Using Docker
docker compose up -d


Qdrant will be available at:

http://localhost:6335


Verification:

http://localhost:6335/collections


Expected response:

{
  "result": [],
  "status": "ok"
}

# Data Preparation

Place textbook PDFs in the following structure:

data/textbooks/
├── class8_science/
│   ├── chapter1.pdf
│   ├── chapter2.pdf
├── class9_science/
├── class10_science/


# Rules:

One PDF corresponds to one chapter

Page numbers are treated as ground truth

Execution Order

Follow the steps below in sequence.

Step 1: Extract Text and Images
python extraction/extract_text.py
python extraction/extract_images.py

Step 2: Create Qdrant Collections
python ingestion/create_collections.py


This creates:

text_pages collection (384-dimensional vectors)

page_images collection (512-dimensional vectors)

Step 3: Ingest Text and Image Data
python ingestion/ingest_text.py
python ingestion/ingest_images.py

Step 4: Run the CLI Application
python cli/main_cli.py

Using the System
Text-Based Query
Ask: What is matter?

Image-Based Query
Ask: Explain the frying image in chapter 1

Voice Input

Select speech input when prompted

Speak clearly into the microphone

Speech-to-Text Setup (Offline)

This project uses Vosk for offline speech recognition.

Model Download

Download the following model:

vosk-model-en-us-0.22-lgraph


From:

https://alphacephei.com/vosk/models

Folder Placement

Extract the model and place it at:

speech/vosk-model/


Expected structure:

speech/
├── speech_to_text.py
├── vosk-model/
│   ├── am/
│   ├── conf/
│   ├── graph/
│   └── ivector/

# Role of Qdrant in the System

Qdrant is used as the vector database to enable:

Fast similarity search over large embedding collections

Separate collections for text and image embeddings

Metadata-based filtering (class, subject, chapter, page)

Scalable and reproducible storage via Docker

Accurate retrieval of page-grounded context for RAG

These features are essential for supporting multimodal, multi-class textbook retrieval.

# Notes and Limitations

All answers are restricted to textbook content

Speech recognition accuracy depends on microphone quality

Large datasets may require additional ingestion time

All scripts must be executed from the project root directory
