## Small Fine-Tuned LLM for IT Ticket Triage & Automation
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)
[![LoRA](https://img.shields.io/badge/LoRA%20%2F%20QLoRA-supported-6f42c1.svg?style=flat-square)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kangarroar/LLM-Ticket-Triage/blob/main/Google_Colab_Training.ipynb)
[![Deployment](https://img.shields.io/badge/deployment-self--hosted-lightgrey.svg?style=flat-square)](#)

This project is a **practical framework** for turning **raw IT issue text** into **strict, schema-constrained JSON tickets** for ITSM tools like Zoho, Jira, and ManageEngine.

The core idea: use a **small fine-tuned LLM** (e.g. Qwen 2.5 1.5B, TinyLLaMA) to do closed-domain ticket structuring, not open-ended chat. The model maps:

- Free text > summary, category, subcategory, priority
- Assignment group routing
- Request type classification (Incident vs. Service Request)

Current Strategy: **Implicit Training + Explicit Inference**.
- **Training**: Teaches the model the *behavior* of extracting and classifying data using varied, natural language instructions.
- **Inference**: Enforces *compliance* by injecting a strict JSON schema (with Enums) into the system prompt.

---

## Why Small Fine-Tuned Models

- **Repetitive, label-constrained task** 
- **Cheaper & faster** than GPT-class APIs
- **More deterministic** and easier to validate against a fixed JSON schema
- Inference can run on **~4 GB VRAM** GPUs with QLoRA/4b-quantization

The model learns **domain-specific categorization** (e.g., "VPN issue" = "Network"), while the inference prompt guarantees the **JSON structure**.

---

## Workflow & Architecture

**1. Generate Data (Synthetic)**
Use `generating/synthetic_data_generation.py` to create thousands of varied IT tickets. It rotates instructions to prevent overfitting.
*For real deployment the dataset should be composed by real tickets already categorized and structured!*

**2. Train (Colab)**
Fine-tune the model using **QLoRA** on Google Colab (free tier supported).
- **Interactive Forms**: Configure rank, alpha, learning rate, and epochs via UI.
- **TensorBoard**: integrated for real-time loss monitoring.
- **Auto-Save**: Adapters saved to Google Drive or downloaded as Zip.
*Config file for local training is optimized for a 1650 Mobile GPU.*

**3. Deploy & Infer**
The notebook includes an **inference test** cell that loads the adapter and strictly forces the output schema:
```json
{
  "summary": "...",
  "category": "Hardware",
  "subcategory": "Other",
  "priority": "High",
  ...
}
```

---

## Repo Structure (Overview)

- `Google_Colab_Training.ipynb` – **Main Entrypoint**. End-to-end training/inference notebook.
- `configs/` – Model and schema config stubs.
- `training/` – Training scripts (`train_lora.py`) and synthetic data tools.
- `inference/` – Prediction engine and schema definitions.
- `data/` – `processed/` directory for JSONL datasets.
- `scripts/` – Utility scripts (e.g., `start_tensorboard.py`).
- `tests/` – Simple smoke tests.

---

## What’s Included vs. Not Included

Included:

- **End-to-End Colab Notebook** (Data Gen -> Train -> Monitor -> Save)
- **Synthetic Data Generator** (Typos, Spanish mix)
- **Schema-First Logic**
- Basic project layout ready for extension

Not included:

- Real IT ticket data.
- Pre-trained model weights.


This repository provides a production-oriented implementation designed to be deployed and fine-tuned **by the user** within their own environment, using their own ticket data, schemas, and ITSM systems.
