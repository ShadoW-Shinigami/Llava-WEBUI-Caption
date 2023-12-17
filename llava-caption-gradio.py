import json
import re
import uuid
import os
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import gradio as gr
from gradio.components import Textbox, Radio, Dataframe
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

disable_torch_init()
torch.manual_seed(1234)

# Load model and other necessary components
MODEL = "liuhaotian/llava-v1.5-13b"
model_name = get_model_name_from_path(MODEL)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL, model_base=None, model_name=model_name, device="cuda"
)

def get_extension_from_url(url):
    """
    Extract the file extension from the given URL.
    """
    parsed_url = urlparse(url)
    path = Path(parsed_url.path)
    return path.suffix

def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def process_image(image):
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)

def create_prompt(prompt: str):
    conv = conv_templates["llava_v0"].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv

def ask_image(image: Image, prompt: str):
    image_tensor = process_image(image)
    prompt, conv = create_prompt(prompt)
    input_ids = (
        tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    return tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

def find_image_urls(data, url_pattern=re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|jpeg|png|webp)')):
    """
    Recursively search for image URLs in a JSON object.
    """
    if isinstance(data, list):
        for item in data:
            for url in find_image_urls(item, url_pattern):
                yield url
    elif isinstance(data, dict):
        for value in data.values():
            for url in find_image_urls(value, url_pattern):
                yield url
    elif isinstance(data, str) and url_pattern.match(data):
        yield data

def gradio_interface(directory_path, prompt, exist):
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    captions = []
    
    # Check for images.json and process it
    json_path = os.path.join(directory_path, 'images.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            image_urls = list(find_image_urls(data))
            for url in image_urls:
                try:
                    # Generate a unique filename for each image with the correct extension
                    extension = get_extension_from_url(url) or '.jpg'  # Default to .jpg if no extension is found
                    unique_filename = str(uuid.uuid4()) + extension
                    unique_filepath = os.path.join(directory_path, unique_filename)
                    response = requests.get(url)
                    with open(unique_filepath, 'wb') as img_file:
                        img_file.write(response.content)
                    image_paths.append(unique_filepath)
                except Exception as e:
                    captions.append((url, f"Error downloading {url}: {e}"))

    # Process each image path with tqdm progress tracker
    for im_path in tqdm(image_paths, desc="Captioning Images", unit="image"):
        base_name = os.path.splitext(os.path.basename(im_path))[0]
        caption_path = os.path.join(directory_path, base_name + '.txt')

        # Handling existing files
        if os.path.exists(caption_path) and exist == 'skip':
            captions.append((base_name, "Skipped existing caption"))
            continue
        elif os.path.exists(caption_path) and exist == 'add':
            mode = 'a'
        else:
            mode = 'w'

        # Image captioning
        try:
            im = load_image(im_path)
            result = ask_image(im, prompt)

            # Writing to a text file
            with open(caption_path, mode) as file:
                if mode == 'a':
                    file.write("\n")
                file.write(result)

            captions.append((base_name, result))
        except Exception as e:
            captions.append((base_name, f"Error processing {im_path}: {e}"))

    return captions

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        Textbox(label="Directory Path"),
        Textbox(default="describe this image in detail", label="Captioning Prompt"),
        Radio(["skip", "replace", "add"], label="Existing Caption Action", default="skip")
    ],
    outputs=[
        Dataframe(type="pandas", headers=["Image", "Caption"], label="Captions")
    ],
    title="Image Captioning",
    description="Generate captions for images in a specified directory."
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()
