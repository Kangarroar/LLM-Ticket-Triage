## Clasificador de Tickets LLM

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)
[![LoRA](https://img.shields.io/badge/LoRA%20%2F%20QLoRA-supported-6f42c1.svg?style=flat-square)](#)
[![Deployment](https://img.shields.io/badge/deployment-self--hosted-lightgrey.svg?style=flat-square)](#)

Este proyecto es un **framework práctico** para convertir **texto bruto de incidencias de IT** en **tickets JSON estrictamente restringidos por esquema** para herramientas Help Desk como Zoho, Jira y ManageEngine.

La idea central: usar un **LLM pequeño* (por ejemplo, Qwen 2.5 1.5B, TinyLLaMA, Phi-2 vía QLoRA) para **estructuración de tickets en dominio cerrado**, no para chat abierto. El modelo mapea:

* Texto libre > resumen, categoría, subcategoría, aplicación
* Urgencia, impacto, grupo de asignación
* Puntuación de confianza para permitir anulación humana segura

---

## Por qué modelos pequeños

* **Tarea repetitiva y con etiquetas restringidas**
* **Más barato y rápido** que APIs GPT
* **Más determinista** y más fácil de validar contra un esquema JSON fijo
* La inferencia puede ejecutarse en GPUs con **~4 GB de VRAM** usando QLoRA / quant de 4 bits

El modelo aprende **mapeo de esquemas y límites de etiquetas**, no lenguaje general.

---

## Arquitectura de alto nivel

**Crear el ticket primero, enriquecer después:**

1. El usuario envía un ticket en la herramienta ITSM.
2. El ticket se crea inmediatamente con los campos mínimos requeridos.
3. Un worker en segundo plano (polling, API, tarea o webhook) recoge los nuevos tickets.
4. El LLM genera JSON estructurado + puntuación de confianza.
5. El ticket se actualiza; los resultados de baja confianza pueden derivar a triaje humano. (Sin pérdida de tickets, reintentos sencillos)

---

## Estructura del repositorio (visión general)

* `configs/` – Stubs de configuración del modelo y del esquema (por ejemplo, ajustes de QLoRA, esquema de tickets).
* `training/` – Punto de entrada de entrenamiento QLoRA, generador de datos sintéticos, wrappers de datasets.
* `inference/` – Motor de predicción, esquemas, post-procesado, app FastAPI, worker.
* `itsm_integrations/` – Stubs de clientes para Zoho, Jira y ManageEngine + bucle de polling/webhook.
* `data/` – Directorios de datos `raw/` y `processed/`.
* `scripts/` – Scripts de utilidad para configuración del entorno y exportación de adaptadores.
* `tests/` – Tests básicos de humo para esquemas y el motor de predicción dummy.

---

## Qué está incluido y qué no

**Incluido:**

* Scripts de entrenamiento y código de inferencia
* **Stubs de esquemas** de tickets y ejemplos de entrada/salida
* **Stubs de integración ITSM**
* Estructura básica del proyecto lista para extender

**No incluido:**

* Datos reales de tickets de IT
* Modelo entrenado y adaptadores

Este repositorio proporciona una implementación orientada a producción, diseñada para ser desplegada y afinada **por el usuario** dentro de su propio entorno, usando sus propios datos de tickets, esquemas y sistemas ITSM.
