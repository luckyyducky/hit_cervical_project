
import torch
import torch.nn.functional as F
import yaml
import argparse
import os
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt

# We need to import the model definition and the specific transform builder
from core.model import get_model
from core.datasets import build_transform  # <-- CORRECTED IMPORT

# A helper function to load the configuration from the YAML file
def get_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # --- 1. SETUP AND ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description='Generate Saliency and JSON for a single image.')
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
    # We create a dummy args object just for the build_transform function
    class DummyArgs:
        input_size = 224
        eval_crop_ratio = 0.875
        
    # Call build_transform directly, which is what we wanted all along
    transform = build_transform(is_train=False, args=DummyArgs()) # <-- CORRECTED FUNCTION CALL

    image = Image.open(args.data_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # --- 4. EXTRACTING ATTENTION SCORES ---
    print("--- Extracting attention scores ---")
    
    attention_scores = None
    
    def attention_hook(module, input, output):
        nonlocal attention_scores
        attention_scores = output[1].detach()

    # Find the last attention layer in the model to attach the hook
    # The model structure is encoder -> layers (Sequential) -> block_x (Sequential) -> encoder_layer_y (EncoderBlockHiT) -> self_attention
    last_block = model.encoder.layers[-1]
    last_encoder_layer = last_block[-1]
    last_attention_layer = last_encoder_layer.self_attention
        
    handle = last_attention_layer.register_forward_hook(attention_hook)

    # --- 5. RUNNING INFERENCE ---
    print("--- Running model inference ---")
    with torch.no_grad():
        logits = model(image_tensor, get_additional_outputs=True)
        
    handle.remove()

    # --- 6. PROCESSING THE OUTPUT ---
    print("--- Processing and saving results ---")
    probabilities = F.softmax(logits, dim=1)[0]
    confidence, predicted_idx = torch.max(probabilities, 0)
    
    patch_importance = attention_scores.mean(dim=1).squeeze(0).cpu().numpy()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # --- 7. GENERATING JSON OUTPUT ---
    patch_importance_dict = {f"patch_{i}": float(score) for i, score in enumerate(patch_importance)}
    
    sorted_patches = sorted(patch_importance_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Class names for SipakaMed dataset
    class_names = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

    output_json = {
        "image_file": os.path.basename(args.data_path),
        "predicted_class": class_names[predicted_idx.item()],
        "confidence_score": round(confidence.item(), 4),
        "patch_importance": {k: round(v, 4) for k, v in sorted_patches}
    }

    json_filename = os.path.splitext(os.path.basename(args.data_path))[0] + ".json"
    json_filepath = os.path.join(args.output_path, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(output_json, f, indent=4)
    print(f"JSON output saved to: {json_filepath}")
    
    # --- 8. GENERATING SALIENCY MAP ---
    num_patches_side = int(np.sqrt(len(patch_importance)))
    saliency_map = patch_importance.reshape(num_patches_side, num_patches_side)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # Use nearest neighbor interpolation to make the patch blocks sharp
    plt.imshow(saliency_map, cmap='viridis', alpha=0.6, extent=(0, image.width, image.height, 0), interpolation='nearest')
    plt.title(f"Predicted: {class_names[predicted_idx.item()]} ({confidence.item():.2f})")
    plt.axis('off')
    
    saliency_filename = os.path.splitext(os.path.basename(args.data_path))[0] + "_saliency.png"
    saliency_filepath = os.path.join(args.output_path, saliency_filename)
    plt.savefig(saliency_filepath, bbox_inches='tight', pad_inches=0)
    print(f"Saliency map saved to: {saliency_filepath}")


if __name__ == '__main__':
    main()
