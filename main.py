import os
import asyncio
import concurrent.futures

from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load the checkpoint path from environment variable, with default
checkpoint = os.getenv("CHECKPOINT", "Qwen/Qwen2.5-VL-3B-Instruct")

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    checkpoint,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
)

app = FastAPI()

# Create a thread pool executor for blocking inference operations.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@app.get("/")
def read_root():
    return {"message": "API is live. Use the /predict endpoint."}

@app.get("/predict")
async def predict(image_url: str = Query(...), prompt: str = Query(...)):
    """
    Offload the blocking model inference to a thread so that the event loop remains responsive.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_prediction, image_url, prompt)
    return {"response": result}

# New: Define a request model for predict POST.
class PredictRequest(BaseModel):
    image_url: str
    prompt: str

# New: Add POST endpoint at /predict.
@app.post("/predict")
async def predict_post(request: PredictRequest):
    """
    Offload the blocking model inference to a thread (POST endpoint version).
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_prediction, request.image_url, request.prompt)
    return {"response": result}

def process_prediction(image_url: str, prompt: str) -> str:
    # Construct the message structure as required by the processor
    messages = [
        {"role": "system", "content": "You are a helpful assistant with vision abilities."},
        {"role": "user", "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": prompt}
        ]},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # Trim the generated ids if needed (assuming the processor's input_ids field exists)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts[0]

# New: Define a request model for chat POST which accepts raw messages.
class ChatRequest(BaseModel):
    messages: list

# New: Add POST endpoint at /chat.
@app.post("/chat")
async def chat_post(request: ChatRequest):
    """
    Offload the blocking model inference using raw messages input.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_chat, request.messages)
    return {"response": result}

def process_chat(messages: list) -> str:
    # Process the raw messages passed to the endpoint.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts[0]

