# **Product Requirements Document (PRD)**

**Project:** Voice Note Taking System with LLM-Enhanced Chunking  
**Version:** 1.1  
**Date:** 2025-02-25

---

## **1\. Overview**

This project will build an end-to-end voice note taking system that captures audio recordings, transcribes them, and automatically segments the transcription into atomic “chunks” of information. The output is stored as Markdown (with YAML metadata) so that it’s both human-readable (e.g., in Obsidian or similar markdown-based systems) and machine-queriable by an LLM-based retrieval system. Additionally, the system will integrate results from both vector and graph databases and include a user interface for browsing the information graphically.

---

## **2\. Objectives**

* **Capture and Transcription:**  
  * Record and process audio notes.  
  * Evaluate multiple transcription options, including locally hosted solutions (e.g., Whisper deployed locally) and API-based solutions (e.g., Deepgram, OpenAI Whisper API).  
* **Atomic Chunking:**  
  * Automatically segment transcriptions into atomic “chunks” based on information content—not just token length.  
  * Ensure that each chunk represents a single atomic unit of information (e.g., a fact or statement).  
* **Dual Storage:**  
  * Export chunks as Markdown files with YAML front matter.  
  * Ingest chunks into both a vector database and a graph database, with the ability to combine results from both during retrieval.  
* **LLM Integration:**  
  * Leverage a retrieval-augmented generation (RAG) pipeline that uses the stored chunks to generate responses.  
* **User Interface:**  
  * Provide human access via Obsidian-compatible Markdown files.  
  * Develop a custom UI that includes a graph viewer for browsing interlinked chunks.  
* **Scalability & Optimization:**  
  * Initially target a low volume (a few notes per day for a single user) with a design that can scale if needed.  
  * Include performance metrics and logging for future scaling considerations.

---

## **3\. System Architecture**

### **3.1 High-Level Workflow**

1. **Audio Capture & Transcription**  
   * **Input:** Audio files in WAV, M4A, and MP3 formats.  
   * **Transcription:**  
     * Evaluate both locally hosted options (e.g., Whisper) and API-based solutions (e.g., Deepgram, Whisper API).  
     * Output a full transcript with detailed timestamps for each sentence or segment.  
2. **Pre-Processing & Chunking**  
   * **Chunking Strategy:**  
     * Use a sliding window approach with overlaps to ensure no loss of context.  
     * Chunking is driven by the information content and atomicity rather than token count.  
   * **Two-Pass Extraction:**  
     * **First Pass:** Extract atomic facts from each transcript window.  
     * **Second Pass:** Use a Q\&A prompt to capture any additional details.  
   * **Post-Processing:**  
     * Deduplicate overlapping chunks.  
     * Each chunk includes metadata such as unique ID, timestamp range, and pointers to related chunks.  
3. **Markdown Export with Metadata**  
   * Export each chunk as a separate Markdown (.md) file.  
   * Use YAML front matter to store metadata (e.g., `id`, `source_file`, `timestamp_range`, `tags`, `links`, and summary).  
   * Replace ambiguous pronouns with explicit references to ensure that each chunk is self-contained.  
4. **Database Ingestion**  
   * **Vector Database:**  
     * Use libraries like FAISS, Milvus, Pinecone, Weaviate, or Chroma to embed and store chunks.  
   * **Graph Database:**  
     * Ingest chunks into a graph database (e.g., Neo4j, TypeDB) where nodes represent chunks and edges represent relationships.  
   * **Result Combination:**  
     * The system will be designed to automatically combine and present results from both the vector and graph databases.  
5. **Query & Retrieval**  
   * **LLM Retrieval Pipeline:**  
     * Embed user queries to retrieve top-k relevant chunks.  
     * Feed the retrieved chunks back into an LLM for context-rich responses.  
6. **User Interface & Visualization**  
   * **Markdown-Based Access:**  
     * Allow users to browse and edit chunks using markdown tools like Obsidian.  
   * **Custom UI:**  
     * Develop a UI component featuring a graph viewer to navigate the network of interlinked chunks.

---

## **4\. Detailed Functional Requirements**

### **4.1 Audio Capture and Transcription**

* **Supported Audio Formats:** WAV, M4A, and MP3.  
* **Transcription Options:**  
  * Evaluate multiple solutions:  
    * **Locally Hosted:** e.g., OpenAI Whisper deployed on-premise.  
    * **API-Based:** e.g., Deepgram, Whisper API.  
* **Output:** Full transcript with timestamps per sentence/segment.

### **4.2 Chunking Process**

#### **4.2.1 Phase 1: Initial Implementation**

* **Chunking:**  
  * Segment the transcript into overlapping windows (e.g., 500–1,000 tokens with 50–100 token overlaps).  
  * Extract atomic facts using a first-pass prompt.  
  * Generate a unique ID for each chunk.  
* **Markdown Export:**  
  * Create Markdown files with YAML front matter including metadata (e.g., `id`, `source_file`, `timestamp_range`).

#### **4.2.2 Phase 2: Chunking Optimization**

* **Enhanced Extraction:**  
  * **Two-Pass Extraction:**  
    * **First Pass:** List atomic facts from each window.  
    * **Second Pass:** Follow up with a Q\&A to capture any missing details.  
  * **Deduplication:**  
    * Merge overlapping chunks into a single master list, ensuring each atomic fact is unique.  
* **Automation Files:**  
  * Develop `chunking_v2.py` implementing the advanced process.  
  * Create a test suite (`test_chunking_v2.py`) to cover various edge cases, including overlapping windows and ambiguous pronoun resolution.

#### **4.2.3 Phase 3: Further Optimizations**

* **Enhanced Prompts:**  
  * Adjust prompts to handle tense variations (e.g., present versus past).  
  * Ensure each chunk is self-contained by replacing ambiguous pronouns with specific names or entities.  
* **Tag & Topic Identification:**  
  * Implement a dedicated pass for identifying tags and topics for each chunk.  
* **Connection Pass:**  
  * Automatically identify and store relationships between chunks.  
* **Metrics Logging:**  
  * Log elapsed time and estimated time until next process completion.  
  * Include final summary metrics (e.g., number of API calls, tokens processed).  
* **Transcript Linking & Context:**  
  * Each chunk should maintain a pointer to its specific transcript segment.  
  * Add contextual metadata about the overall session (e.g., session title, participants, recording location).  
* **JSON Parsing:**  
  * Address issues with commented lines in JSON returns by applying post-processing or sanitization.  
* **Instruct Mistral Integration:**  
  * Confirm compatibility with the instruct version of Mistral for improved prompting flows.

---

## **5\. Non-Functional Requirements**

* **Performance:**  
  * Must handle transcripts efficiently, even with overlapping windows.  
  * Retrieval operations should operate in near real-time.  
* **Scalability:**  
  * Designed for low usage initially (a few notes per day) but with scalability in mind.  
* **Reliability:**  
  * Robust error handling during transcription, chunking, and database ingestion.  
  * Detailed and persistent logging for metrics and process statuses.  
* **Maintainability:**  
  * Modular, well-documented code with comprehensive unit tests (especially for `chunking_v2.py`).  
* **Interoperability:**  
  * Markdown output compatible with Obsidian and other markdown-based systems.  
* **Security:**  
  * Secure storage and handling of audio, transcriptions, and metadata in compliance with data protection policies.

---

## **6\. Implementation Plan & Suggestions**

### **6.1 Fast Implementation & Optimizations**

* **Transcription Service Evaluation:**  
  * Create modular integrations for both locally hosted (e.g., Whisper) and API-based transcription services (e.g., Deepgram, Whisper API) to allow for A/B testing.  
* **Modular Architecture:**  
  * Clearly separate the modules for transcription, chunking, database ingestion, and UI, allowing for parallel development and future scalability.  
* **Rapid Prototyping:**  
  * Start with a minimal viable product (MVP) that includes audio capture, basic transcription, and simple chunking before integrating advanced features.  
* **Prompt Engineering:**  
  * Iteratively refine prompts on test transcripts to ensure atomic facts are correctly extracted.  
* **Metrics Dashboard:**  
  * Build a lightweight dashboard for real-time monitoring of process metrics (elapsed time, token counts, API calls).

### **6.2 Questions & Clarifications Addressed**

1. **Transcription Service Selection:**  
   * Evaluate multiple transcription options (both local and API-based).  
2. **Languages & Formats:**  
   * Initially support English language with WAV, M4A, and MP3 audio files.  
3. **Chunking Granularity:**  
   * Chunks will be generated based on atomic information content rather than fixed token lengths.  
4. **Database Integration:**  
   * Design the system to automatically combine retrieval results from both vector and graph databases.  
5. **User Interface:**  
   * In addition to Obsidian integration, develop a custom UI featuring a graph viewer for browsing and editing chunks.  
6. **Usage Levels:**  
   * Assume low-volume usage (a few notes per day for a single user) with scalability considerations for future growth.

---

## **7\. Deliverables & Milestones**

* **Phase 1 Deliverables:**  
  * Integration with multiple transcription services.  
  * Basic transcription-to-chunk pipeline with Markdown export.  
  * Ingestion of chunks into a vector database with initial retrieval capabilities.  
* **Phase 2 Deliverables:**  
  * Advanced chunking module (`chunking_v2.py`) implementing the sliding window and two-pass extraction.  
  * Comprehensive testing suite (`test_chunking_v2.py`).  
* **Phase 3 Deliverables:**  
  * Enhanced prompt adjustments (tense handling, pronoun resolution).  
  * Tag/topic identification and connection passes.  
  * Metrics logging system with performance and token tracking.  
  * Integration with both vector and graph databases and automated result combination.  
  * Custom UI with graph viewer for browsing chunks.  
* **Documentation:**  
  * Developer documentation for all modules.  
  * API documentation for integration points and process flows.

---

## **8\. Open Issues & Future Considerations**

* **Ambiguous Pronoun Resolution:**  
  * Investigate NLP models or rules-based solutions to resolve ambiguous pronouns in real time.  
* **JSON Parsing & Sanitization:**  
  * Implement sanitization steps to handle JSON responses with commented lines.  
* **Real-Time vs. Batch Processing:**  
  * Decide whether transcription and chunking occur in real time or via scheduled batch processing.  
* **Scaling Projections:**  
  * Monitor usage to adjust performance and scaling plans as needed.

---

