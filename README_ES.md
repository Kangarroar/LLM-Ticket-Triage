## Clasificador de tickets IT

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)
[![LoRA](https://img.shields.io/badge/LoRA%20%2F%20QLoRA-supported-6f42c1.svg?style=flat-square)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kangarroar/LLM-Ticket-Triage/blob/main/Google_Colab_Training.ipynb)
[![Deployment](https://img.shields.io/badge/deployment-self--hosted-lightgrey.svg?style=flat-square)](#)

Este proyecto es un **framework práctico** para convertir **texto crudo de incidencias de IT** en **tickets JSON estrictos y restringidos por esquema** para herramientas ITSM como Zoho, Jira y ManageEngine.

La idea central: usar un **LLM pequeño ajustado (fine-tuned)** (p. ej., Qwen 2.5 1.5B, TinyLLaMA) para estructuración de tickets en un dominio cerrado, no para chat abierto. El modelo mapea:

* Texto libre → resumen, categoría, subcategoría, prioridad
* Enrutamiento por grupo de asignación
* Clasificación del tipo de solicitud (Incidente vs. Solicitud de Servicio)

Estrategia actual: **Entrenamiento Implícito + Inferencia Explícita**.

* **Entrenamiento**: enseña al modelo el *comportamiento* de extraer y clasificar datos usando instrucciones variadas en lenguaje natural.
* **Inferencia**: fuerza el *cumplimiento* inyectando un esquema JSON estricto (con Enums) en el prompt del sistema.

---

## Por Qué Modelos Fine-Tuned

* **Tarea repetitiva y restringida por etiquetas**
* **Más baratos y rápidos** que las APIs de clase GPT
* **Más deterministas** y más fáciles de validar contra un esquema JSON fijo
* La inferencia puede ejecutarse en GPUs con **~4 GB de VRAM** usando QLoRA / cuantización a 4 bits

El modelo aprende **categorización específica del dominio** (p. ej., “problema de VPN” = “Red”), mientras que el prompt de inferencia garantiza la **estructura JSON**.

---

## Flujo de Trabajo y Arquitectura

**1. Generar Datos (Sintéticos)**
Usa `generating/synthetic_data_generation.py` para crear miles de tickets de IT variados. Rota las instrucciones para evitar el sobreajuste.
*Para un despliegue real, el dataset debería componerse a partir de tickets reales ya categorizados y estructurados.*

**2. Entrenar (Colab)**
Ajusta el modelo usando **QLoRA** en Google Colab (compatible con el tier gratuito).

* **Formularios Interactivos**: configura rank, alpha, learning rate y épocas vía UI.
* **TensorBoard**: integrado para monitorización de la pérdida en tiempo real.
* **Auto-Guardado**: los adaptadores se guardan en Google Drive o se descargan como ZIP.
  *El archivo de configuración para entrenamiento local está optimizado para una GPU 1650 Mobile.*

**3. Desplegar e Inferir**
El notebook incluye una celda de **prueba de inferencia** que carga el adaptador y fuerza estrictamente el esquema de salida:

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

## Estructura del Repositorio (Resumen)

* `Google_Colab_Training.ipynb` – **Punto de Entrada Principal**. Notebook de entrenamiento/inferencia de extremo a extremo.
* `configs/` – Stubs de configuración del modelo y del esquema.
* `training/` – Scripts de entrenamiento (`train_lora.py`) y herramientas de datos sintéticos.
* `inference/` – Motor de predicción y definiciones de esquema.
* `data/` – Directorio `processed/` para datasets en JSONL.
* `scripts/` – Scripts de utilidad (p. ej., `start_tensorboard.py`).
* `tests/` – Pruebas básicas (smoke tests).

---

## Qué Está Incluido vs. Qué No Está Incluido

Incluido:

* **Notebook de Colab de Extremo a Extremo** (Generación de Datos → Entrenamiento → Monitorización → Guardado)
* **Generador de Datos Sintéticos** (errores tipográficos, mezcla con español)
* **Schema-First**
* Estructura básica del proyecto lista para extenderse

No incluido:

* Datos reales de tickets de IT.
* Pesos de modelos preentrenados.

Este repositorio proporciona una implementación orientada a producción, diseñada para ser desplegada y ajustada **por el usuario** dentro de su propio entorno, utilizando sus propios datos de tickets, esquemas y sistemas ITSM.
