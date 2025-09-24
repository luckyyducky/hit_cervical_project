
import torch
import torch.nn.functional as F
import yaml
import argparse
import os
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt

# Import the necessary functions from the project
from core.model import get_model
from core.datasets import build_transform

def get_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # --- 1. SETUP AND ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description='Generate Layer-wise Saliency for a single image.')
    parser.add_argument('--data-path', required=True, type=str, help='Path to the single image file.')
    parser.add_argument('--weights', required=True, type=str, help='Path to the fine-tuned model checkpoint.')
    parser.add_argument('--config-file', required=True, type=str, help='Path to the model config file.')
    parser.add_argument('--output-path', required=True, type=str, help='Directory to save the outputs.')
    parser.add_argument('--num-classes', required=True, type=int, help='Number of classes in the dataset.')
    args = parser.parse_args()

    print("--- Loading configuration ---")
    config = get_config(args.config_file)
    config['model']['params']['num_classes'] = args.num_classes
    
    # --- 2. MODEL LOADING ---
    print("--- Building and loading model ---")
    model_args = config['model']['params']
    model, _ = get_model(config['model']['name'], model_args)
    
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}.")

    # --- 3. IMAGE LOADING AND TRANSFORMATION ---
    print("--- Preparing image ---")
    class DummyArgs:
        input_size = 224
        eval_crop_ratio = 0.875
    transform = build_transform(is_train=False, args=DummyArgs())

    image = Image.open(args.data_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # --- 4. ATTACHING HOOKS TO CAPTURE LAYER OUTPUTS ---
    print("--- Attaching hooks to all encoder layers ---")
    
    intermediate_class_tokens = []
    
    # This hook function will be called after each layer's forward pass
    def layer_output_hook(module, input, output):
        # The output of EncoderBlockHiT is (input, class_token, other_outputs)
        # We want to capture the class_token, which is the second element.
        intermediate_class_tokens.append(output[1].detach())

    # Iterate through the complex nested structure of the SpatialEncoder to find every layer
    for block in model.encoder.layers:
        for module_name, module in block.named_modules():
            # We are looking for the specific encoder block class
            if 'EncoderBlockHiT' in str(type(module)):
                module.register_forward_hook(layer_output_hook)

    # --- 5. RUNNING INFERENCE ---
    print("--- Running model inference to trigger hooks ---")
    with torch.no_grad():
        logits = model(image_tensor)
        
    # Get the final prediction details
    final_probabilities = F.softmax(logits, dim=1)[0]
    final_confidence, final_predicted_idx = torch.max(final_probabilities, 0)

    # --- 6. ANALYZING INTERMEDIATE OUTPUTS ---
    print("--- Analyzing intermediate layer outputs ---")
    layer_confidences = []
    with torch.no_grad():
        for token in intermediate_class_tokens:
            # We must pass the intermediate token through the final LayerNorm
            # before feeding it to the classification head for a fair comparison.
            normalized_token = model.encoder.ln(token)
            
            # Get the logits this layer would have produced
            layer_logits = model.heads(normalized_token.squeeze(1))
            layer_probabilities = F.softmax(layer_logits, dim=1)[0]
            
            # Get the confidence of the *final* predicted class at this layer
            confidence_at_this_layer = layer_probabilities[final_predicted_idx].item()
            layer_confidences.append(confidence_at_this_layer)
    
    # Create the directory for outputs if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # --- 7. GENERATING JSON OUTPUT ---
    class_names = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
    final_predicted_class_name = class_names[final_predicted_idx.item()]

    output_json = {
        "image_file": os.path.basename(args.data_path),
        "final_predicted_class": final_predicted_class_name,
        "final_confidence_score": round(final_confidence.item(), 4),
        "layer_confidence_progression": {f"layer_{i+1}": round(conf, 4) for i, conf in enumerate(layer_confidences)}
    }
    
    json_filename = os.path.splitext(os.path.basename(args.data_path))[0] + "_layer_saliency.json"
    json_filepath = os.path.join(args.output_path, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(output_json, f, indent=4)
    print(f"Layer-wise JSON output saved to: {json_filepath}")
    
    # --- 8. GENERATING BAR CHART VISUALIZATION ---
    num_layers = len(layer_confidences)
    layer_indices = range(1, num_layers + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(layer_indices, layer_confidences, color='skyblue')
    ax.set_xticks(layer_indices)
    ax.set_xlabel('Transformer Layer Number', fontsize=12)
    ax.set_ylabel(f'Confidence in Final Prediction ("{final_predicted_class_name}")', fontsize=12)
    ax.set_title('Model Confidence Progression Through Layers', fontsize=14, weight='bold')
    ax.set_ylim(0, 1)
    
    # Add confidence values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

    saliency_filename = os.path.splitext(os.path.basename(args.data_path))[0] + "_layer_saliency.png"
    saliency_filepath = os.path.join(args.output_path, saliency_filename)
    plt.savefig(saliency_filepath, bbox_inches='tight', dpi=150)
    print(f"Layer-wise saliency chart saved to: {saliency_filepath}")

if __name__ == '__main__':
    main()
