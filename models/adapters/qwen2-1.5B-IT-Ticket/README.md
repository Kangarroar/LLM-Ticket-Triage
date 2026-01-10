---
base_model: Qwen/Qwen2.5-1.5B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B
- lora
- transformers
---

# Model Card for Qwen2.5-1.5B-IT-Ticket

This was trained for 3 epoches, 5000 tickets, with lr of 1e-5.

- PEFT 0.18.0