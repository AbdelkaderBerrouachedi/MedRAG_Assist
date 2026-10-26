# MedRAG-Assist: Modular Retrieval-Augmented Generation for Ambient Clinical Consultation Summarization

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Target Venue](https://img.shields.io/badge/Journal-Computers%20in%20Biology%20and%20Medicine%20(Elsevier)-orange.svg)](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine)

Official open-source implementation and replication benchmark for the **MedRAG-Assist** framework: a modular neuro-symbolic retrieval-augmented generation pipeline designed for ambient clinical consultation summarization, evidence grounding, and contraindication audit.

---

## 📋 Architectural Overview

MedRAG-Assist coordinates six modular stages to transform ambient clinical consultations into structured, audit-traceable SOAP reports while mitigating clinical hallucinations:

1. **Phase I: Multimodal Dialogue Ingestion** – Ingestion of ambient audio transcripts, telehealth chat logs, and turn-based speaker diarization.
2. **Phase II: Calibrated Relevance Filtering ($\tau = 0.65$)** – BioBERT binary classifier to discard conversational chitchat, achieving **38.4% token reduction** and preventing context poisoning.
3. **Phase III: Clinical Entity Extraction & UMLS Normalization** – Structured JSON extraction mapped to standardized Unified Medical Language System (UMLS) Concept Unique Identifiers (CUIs).
4. **Phase IV: Dual Hybrid Retrieval & Reciprocal Rank Fusion ($k_0 = 60$)** – Dual dense (BioBERT) and sparse (BM25) search over authoritative clinical literature (PubMed, NICE, WHO, DrugBank), fused via RRF and re-ranked via Cross-Encoder.
5. **Phase V: Cross-Attention Fusion & Projection** – Learned linear projection matrix $W_{\text{proj}} \in \mathbb{R}^{768 \times 4096}$ reconciling 768-d BioBERT representations into 4096-d Llama-3-8B space.
6. **Phase VI: Evidence-Grounded SOAP Generation** – Conditioned Llama-3-8B decoding yielding structured SOAP summaries with fine-grained ALCE-compliant inline citations, reducing Clinically Contradictory Statements (CCS) by **-77.8%** ($1.8 \to 0.4$).

---

## 📂 Repository Structure

```
MedRAG_Assist/
├── demo_sample.json          # Realistic clinical encounter (Cardiology / Respiratory)
├── requirements.txt          # Python dependencies (torch, transformers, rank-bm25)
├── run_pipeline.py           # Master end-to-end execution script (Phases I through VI)
├── LICENSE                   # Apache 2.0 Open Source License
├── README.md                 # System documentation
└── src/
    ├── relevance_filter.py   # Phase II: BioBERT classifier (tau = 0.65)
    ├── umls_extractor.py     # Phase III: Clinical extraction & UMLS CUI linking
    ├── hybrid_retriever.py   # Phase IV: Dense BioBERT + Sparse BM25 + RRF
    ├── cross_attention.py    # Phase V: PyTorch Cross-Attention Fusion Layer
    ├── soap_generator.py     # Phase VI: Conditioned SOAP Generation & Citations
    └── evaluate.py           # Evaluation: ROUGE-L, BERTScore, UMLS-F1, CCS
```

---

## 🚀 Quickstart Installation

```bash
# Clone the repository
git clone https://github.com/AbdelkaderBerrouachedi/MedRAG_Assist.git
cd MedRAG_Assist

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Running the Master Pipeline

To run the full end-to-end clinical summarization on the sample consultation encounter:

```bash
python run_pipeline.py
```

### Sample Output:
```text
================================================================================
        MedRAG-Assist: Evidence-Grounded Clinical Documentation Pipeline
        Target Venue: Computers in Biology and Medicine (Elsevier)
================================================================================
[PHASE I: INGESTION] Processing Encounter: MEDRAG-ENC-042
[PHASE II: RELEVANCE FILTERING] BioBERT Classifier (tau = 0.65)...
  -> Discarded non-medical turns: 2
  -> Token reduction: 17.3%
[PHASE III: EXTRACTION] UMLS CUIs Mapped: Cough (C0010200), Fever (C0015967)
[PHASE IV: HYBRID RETRIEVAL] Top-3 Guidelines Retrieved via RRF (k0 = 60):
  [Rank 1] NICE-NG28-2023: Pneumonia in Adults Diagnosis & Management
  [Rank 2] ESC-CAD-2024: Guidelines on Coronary Syndromes
[PHASE V: CROSS-ATTENTION] Fused Context E_F in R^(8 x 4096)
[PHASE VI: SOAP REPORT GENERATION]
  [S] Progressive cough & fever (3 wks), Penicillin allergy (hives)
  [O] Reported fever 38.6°C, resting respiration normal
  [A] 1. Community-Acquired Pneumonia [NICE-NG28-2023]
  [P] Urgent CXR, CBC, CRP. Clarithromycin 500mg PO BID (Avoid Amoxicillin!)
  Safety Audit: Contraindication Check PASSED | CCS Reduction: -77.8%
================================================================================
```

---

## 📊 Benchmark Evaluation Replication

To inspect and replicate the empirical performance metrics:

```bash
python src/evaluate.py
```

| Model / Architecture | BERTScore | ROUGE-L | UMLS-F1 | CCS $\downarrow$ | Safety Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Vanilla LLM (Llama-3-8B)** | 0.74 | 0.38 | 0.62 | 1.8 | High Hallucination Risk |
| **LLM + Standard Dense RAG** | 0.83 | 0.44 | 0.73 | 0.9 | Moderate Hallucination Risk |
| **Fine-Tuned + Dense RAG + Filter** | 0.87 | 0.48 | 0.78 | 0.5 | Low Risk |
| **MedRAG-Assist (Proposed Framework)** | **0.89** | **0.51** | **0.81** | **0.4** | **Clinically Grounded (-77.8% CCS)** |

---

## ⚖️ License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete terms.
