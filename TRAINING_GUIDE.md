# Training Guide

Training can be done in two ways:

1. Using Google Colab (free tier supported)
2. Using a local machine

## Prepare your dataset

1. Tickets need to be formatted as JSONL files
2. The JSONL file should have the following structure:
```json
{
    "summary": "...",
    "category": "Hardware",
    "subcategory": "Other",
    "priority": "High",
    ...
}
```
How each line on train.jsonl shoud look like is:
```json
{"instruction": "Transform the user message into a structured IT ticket. Output JSON only.", "input": "I saw someone installing games on the office PC.??", "output": "{\"summary\": \"Installing unauthorized software\", \"category\": \"Security\", \"subcategory\": \"Policy Violation\", \"priority\": \"Low\", \"assignment_group\": \"Security / Access\", \"request_type\": \"Incident\"}"}
```
- Instruction: The instruction to be given to the model
- Input: The message of the ticket
- Output: The categorized output of the ticket in JSON format.
You should always prefer to use real tickets data from your Help Desk system for training, it has to be classified already with your internal categories, subcategories, assignment groups and request types; and more than 1000 tickets minimum.

*Most of Help Desk systems allow you to use REST API, where you'll need to use the search endpoint and handle pagination to retrieve all results, there's an example of said script on the repository. configs/itsm_integrations/zoho/

## Choose your base model

Default is Qwen 2.5-1.5B, but you can use any model you want like TinyLlama. Just make sure to change the model name on the training script, it will automatically fetch the model from HuggingFace.

There's no need to use a model with more than 3B parameters for this task, 1B~ will be enough, going higher than 3B will end up in diminishing returns and more resources cost than just using a smaller model.

## Using Google Colab

1. Open the Google Colab notebook
2. Follow the instructions in the notebook
3. Save the model

Specific Colab configs: 
- If using a T4 GPU, always stay on fp16 for best performance
- Don't enable prediction_loss or you will OOM even in a 32gb machine

## Running on local machine

1. Install dependencies (Recommended to use Python 3.11.14)
2. Prepare your dataset
3. Run the training script
4. Save the model

*The Jupyter notebook is just for reference but it should work locally too.*

# Training parameters

- Rank: 8   
- Alpha: 16
These should be enough for most use cases, but you can adjust them if needed.

- Learning rate: 1e-5
- Epochs: 3
Unless you have diverse ticket data, and a lot of them, 3 epochs should be enough to not overfit.

- Max length: 512
You probably won't need 512 max_length (tokens) for this task, even if tickets are really long, since 512 is 2500 characters.

- Warmup steps: 1000
Warmup should not be more than 5% of the total steps, with 3000 tickets and 3 epochs, training is around 600 steps, so warmup should be around 30-50 steps. Bigger datasets will require more warmup steps.