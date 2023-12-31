from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

from dataclasses import dataclass
from typing import List, Optional
from djl_python import Input, Output
#from safetensors.numpy import load_file, save_file

from PIL import Image
import base64
from io import BytesIO
import json
import os

import deepspeed
from transformers import pipeline

@dataclass 
class Config:
    # models can optionally be passed in directly
    caption_model = None

    caption_model_name: Optional[str] = 'Qwen/Qwen-VL-Chat' # use a key from CAPTION_MODELS or None
    caption_offload: bool = False
    
    # interrogator settings
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class QwenVL():
    def __init__(self, config: Config, properties):
        self.history = None
        
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.load_caption_model(properties)
        self.caption_offloaded = True

        # interrogator settings
        device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
    def load_caption_model(self, properties):
        if self.config.caption_model is None and self.config.caption_model_name:
            model_path = self.config.caption_model_name
            if "model_id" in properties:
                model_path = properties["model_id"]
                if any(os.listdir(model_path)):
                    files_in_folder = os.listdir(model_path)
                    print('model path files:')
                    for file in files_in_folder:
                        print(file)
                else:
                    raise ValueError('Please make sure the model artifacts are uploaded to s3')

            print(f'model path: {model_path}')
            # Note: The default behavior now has injection attack prevention off.
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # use bf16
            self.caption_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, fp16=True).eval()
            # use cpu only
            # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
            #if not self.config.caption_offload:
            #    caption_model = caption_model.to(self.config.device)
        else:
            self.caption_model = self.config.caption_model

    def generate_caption(self, images_path: [str], prompt: Optional[str]=None, params: Optional[dict]={}) -> str:
        assert self.caption_model is not None, "No caption model loaded."
        query_params = [{'text': prompt}]
        for image_path in images_path:
            query_params.append({'image': image_path})
        
        if 'reset_history' in params:
            self.history = None
        
        query = self.tokenizer.from_list_format(query_params)
        response, self.history = self.caption_model.chat(self.tokenizer, query=query, history=self.history)
        return response

    """
    def _prepare_caption(self):
        if self.caption_offloaded:
            self.caption_model = self.caption_model.to(self.device)
            self.caption_offloaded = False
    """
with open('./model_name.json', 'rb') as openfile:
    json_object = json.load(openfile)

model_name = json_object.pop('caption_model_name')
config = None
_service = None

def handle(inputs: Input) -> Optional[Output]:
    global config, _service
    if not _service:
        config = Config()
        config.caption_model_name=model_name
        _service = QwenVL(config, inputs.get_properties())
    
    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    
    images_path = []
    if "images" in data:
        encoded_images = data.pop("images")
        
        for idx, encoded_image in enumerate(encoded_images):
            # Using a context manager to ensure the BytesIO stream is properly closed
            with BytesIO(base64.b64decode(encoded_image)) as f:
                with Image.open(f) as input_image:
                    # Convert the image to 'RGB' only if it's not already in 'RGB' mode
                    if input_image.mode != 'RGB':
                        input_image = input_image.convert('RGB')
                    image_path = f'/tmp/.djl.ai/download/input_image_{idx}.jpg'
                    input_image.save(image_path)
                    images_path.append(image_path)
    
    if 'prompt' in data:
        prompt = data.pop("prompt")
    else:
        prompt = 'Describe this photo'
    
    params = data["parameters"] if 'parameters' in data else {}
    generated_text = _service.generate_caption(images_path, prompt, params)

    return Output().add(generated_text)