## Small Fine-Tuned LLM for IT Ticket Triage & Automation

This project is a **practical framework** for turning **raw IT
issue text** into **strict, schema-constrained JSON tickets** for ITSM tools
like Zoho, Jira, and ManageEngine.

The core idea: use a **small fine-tuned LLM** (e.g. Qwen 2.5 1.5B, TinyLLaMA,
Phi-2 via QLoRA) to do closed-domain ticket structuring, not open-ended
chat. The model maps:

- Free text > summary, category, subcategory, application
- Urgency, impact, assignment group
- Confidence score for safe human override

---

## Why Small Fine-Tuned Models

- **Repetitive, label-constrained task** 
- **Cheaper & faster** than GPT-class APIs
- **More deterministic** and easier to validate against a fixed JSON schema
- Inference can run on **~4 GB VRAM** GPUs with QLoRA/4b-quantization

The model learns **schema mapping and label boundaries**, not general language.

---

## High-Level Architecture

**Create ticket first, enrich later:**

1. User submits a ticket in the ITSM tool.
2. Ticket is created immediately with minimal required fields.
3. A background worker (polling, API, task or webhook) picks up new tickets.
4. The LLM generates structured JSON + confidence.
5. The ticket is updated; low-confidence results can fall back to human triage. (No ticket loss, easy retries)

---

## Repo Structure (Overview)

- `configs/` – Model and schema config stubs (e.g. QLoRA settings, ticket schema).
- `training/` – QLoRA training entrypoint, synthetic data generator, dataset wrappers.
- `inference/` – Prediction engine, schemas, post-processing, FastAPI app, worker.
- `itsm_integrations/` – Zoho, Jira, ManageEngine client stubs + polling/webhook loop.
- `data/` – `raw/` and `processed/` data directories
- `scripts/` – Utility scripts for env setup and adapter export.
- `tests/` – Simple smoke tests for schemas and the dummy prediction engine.

---

## What’s Included vs. Not Included

Included:

- Training scripts & inference code
- Ticket **schema stubs** and example IO
- ITSM integration **client stubs**
- Basic project layout ready for extension

Not included:

- Real IT ticket data
- Trained model or adapter weights


This repository provides a production-oriented implementation designed to be deployed and fine-tuned **by the user** within their own environment, using their own ticket data, schemas, and ITSM systems.



