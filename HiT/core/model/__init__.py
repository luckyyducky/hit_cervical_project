
import torch
import timm
from core.model.IT import SpatialInterpretableTransformer
from core.model.utils import interpolate_embeddings

def get_model(model_name, model_args, weights=''):
    args_for_training = {}
    
    if model_name == 'sivit':
        print('Loading Spatial HiT (sivit)')
        model = SpatialInterpretableTransformer(**model_args)
        if weights:
            print(f'Loading weights from "{weights}"')
            sd = torch.load(weights, map_location='cpu')
            
            # The model was likely saved with a 'model.' prefix, remove it
            if 'model' in sd:
                sd = sd['model']
            
            # Interpolate embeddings for different image sizes if necessary
            sd = interpolate_embeddings(image_size=model_args['image_size'],
                                        patch_size=model_args['patch_size'],
                                        model_state=sd,
                                        reset_heads=True,
                                        reset_conv_proj=False)
            model.load_state_dict(sd, strict=False)
    else:
        raise ValueError(f"Unknown model name '{model_name}' specified in config.")

    print('Load done')
    return model, args_for_training
