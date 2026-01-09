import gradio as gr
import yaml
import subprocess
import sys
import os
import threading
import time
from inference.engine import InferenceEngine

# Initialize Engine
engine = InferenceEngine()

CONFIG_PATH = "configs/training_config.yaml"

def load_yaml_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

def save_yaml_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

# Training Logic
def start_training(model_name, epochs, batch_size, lr, lora_r, lora_alpha, lora_dropout):
    config = load_yaml_config()
    
    # Safely update nested keys
    if "model" not in config: config["model"] = {}
    config["model"]["name"] = model_name
    
    if "training" not in config: config["training"] = {}
    config["training"]["num_train_epochs"] = int(epochs)
    config["training"]["per_device_train_batch_size"] = int(batch_size)
    config["training"]["learning_rate"] = float(lr)
    
    if "lora" not in config: config["lora"] = {}
    config["lora"]["r"] = int(lora_r)
    config["lora"]["lora_alpha"] = int(lora_alpha)
    config["lora"]["lora_dropout"] = float(lora_dropout)
    
    save_yaml_config(config)
    
    yield "Starting training\n"
    
    cmd = [sys.executable, "training/train_lora.py"]
    
    # Use Popen to capture output in real-time
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1,
        cwd=os.getcwd()
    )
    
    logs = ""
    
    # Read output line by line
    for line in iter(process.stdout.readline, ''):
        logs += line
        yield logs
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code == 0:
        logs += "\nTraining Completed Successfully"
    else:
        logs += f"\nTraining Failed with return code {return_code}"
    
    yield logs

# Inference Logic
def trigger_load_model(model_name, adapter_path):
    yield "Loading model."
    result = engine.load_model(model_name=model_name, adapter_path=adapter_path)
    yield result

def trigger_inference(text, max_tokens, use_sampling):
    if not engine.model:
        return "Model not loaded", "Error", ""
    
    result_text, status, validation_errors = engine.predict(text, max_tokens=max_tokens, sampling=use_sampling)
    
    # Format validation errors for display
    errors_str = "\n".join([f"- {err}" for err in validation_errors]) if validation_errors else ""
    
    return result_text, status, errors_str

# UI Layout
def create_ui():
    current_config = load_yaml_config()
    
    def_model = current_config.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B")
    def_epochs = current_config.get("training", {}).get("num_train_epochs", 3)
    def_bs = current_config.get("training", {}).get("per_device_train_batch_size", 4)
    def_lr = current_config.get("training", {}).get("learning_rate", 2e-4)
    
    def_lora_r = current_config.get("lora", {}).get("r", 16)
    def_lora_alpha = current_config.get("lora", {}).get("lora_alpha", 32)
    def_lora_dropout = current_config.get("lora", {}).get("lora_dropout", 0.05)

    with gr.Blocks(title="LLM Ticket Triage Dashboard", theme=gr.themes.Default()) as demo:
        gr.Markdown("# LLM Ticket Triage Dashboard")
        
        with gr.Tabs():
            # TRAINING
            with gr.TabItem("Training"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")
                        t_model = gr.Dropdown(
                            choices=["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
                            value=def_model,
                            label="Base Model",
                            allow_custom_value=True
                        )
                        t_epochs = gr.Slider(1, 10, value=def_epochs, step=1, label="Epochs")
                        t_bs = gr.Dropdown([1, 2, 4, 8, 16], value=def_bs, label="Batch Size")
                        t_lr = gr.Number(value=def_lr, label="Learning Rate", precision=6)
                        
                        with gr.Accordion("LoRA Advanced Settings", open=False):
                            t_r = gr.Slider(8, 64, value=def_lora_r, step=8, label="LoRA Rank (r)")
                            t_alpha = gr.Slider(16, 128, value=def_lora_alpha, step=16, label="LoRA Alpha")
                            t_dropout = gr.Slider(0.0, 0.5, value=def_lora_dropout, step=0.01, label="LoRA Dropout")
                            
                        btn_train = gr.Button("Start Training", variant="primary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Training Logs")
                        t_logs = gr.TextArea(label="Process Output", lines=20, max_lines=30, interactive=False)
                
                btn_train.click(
                    start_training,
                    inputs=[t_model, t_epochs, t_bs, t_lr, t_r, t_alpha, t_dropout],
                    outputs=[t_logs]
                )

            # INFERENCE
            with gr.TabItem("Inference"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Loader")
                        i_model = gr.Dropdown(
                            choices=["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
                            value=def_model,
                            label="Base Model",
                            allow_custom_value=True
                        )
                        
                        # Helper to list adapters
                        def get_available_adapters():
                            adapter_root = "./models/adapters"
                            if not os.path.exists(adapter_root):
                                return []
                            return [os.path.join(adapter_root, d) for d in os.listdir(adapter_root) if os.path.isdir(os.path.join(adapter_root, d))]
                        
                        potential_adapters = get_available_adapters()
                        default_adapter = potential_adapters[0] if potential_adapters else "./models/adapters/qwen2-1.5B-IT-Ticket"

                        i_adapter = gr.Dropdown(
                            choices=potential_adapters, 
                            value=default_adapter, 
                            label="Adapter Path", 
                            allow_custom_value=True
                        )
                        btn_load = gr.Button("Load Model", variant="secondary")
                        load_status = gr.Text(label="Status", interactive=False)
                        
                        gr.Markdown("### Test Input")
                        i_text = gr.TextArea(lines=4, placeholder="Type a ticket description here...", label="Ticket Text", value="outlook crash")
                        
                        with gr.Accordion("Inference Settings", open=False):
                            i_max_tokens = gr.Slider(50, 300, value=196, step=1, label="Max New Tokens")
                            i_sampling = gr.Checkbox(value=False, label="Use Sampling")
                        
                        btn_infer = gr.Button("Run Inference", variant="primary")
                        
                    with gr.Column():
                        gr.Markdown("### Results")
                        with gr.Group():
                            o_status = gr.Text(label="Validation Status")
                            o_errors = gr.TextArea(label="Validation Errors", lines=3, interactive=False)
                        o_json = gr.Code(language="json", label="Generated JSON")
                
                btn_load.click(trigger_load_model, inputs=[i_model, i_adapter], outputs=[load_status])
                btn_infer.click(trigger_inference, inputs=[i_text, i_max_tokens, i_sampling], outputs=[o_json, o_status, o_errors])

    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.queue().launch(share=False, server_name="127.0.0.1")
