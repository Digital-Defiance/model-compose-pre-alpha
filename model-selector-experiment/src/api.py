
import fastapi
import torch
import uvicorn
import torch
import time
import contextlib

from pydantic import BaseModel
from typing import Optional, Any
import os
import sys
import torch
import random
import argparse
import numpy as np

from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
from typing import Literal
import redis
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

r = redis.StrictRedis(host='redis', port=6379, db=0, decode_responses=True)

r.set("LOCK", 0)


@contextlib.contextmanager
def redis_lock():
    logger.debug("Waiting for GPU LOCK")
    for _ in range(10):
        if r.get("LOCK") == '0':
            try:
                r.set("LOCK", 1)
                logger.debug("Acquired GPU LOCK")
                yield
                return
            finally:
                r.set("LOCK", 0)
                logger.debug("Released GPU LOCK")
        time.sleep(1)

    logger.error("GPU was not released.")
    raise fastapi.HTTPException(500, "Gpu was not available")


app = fastapi.FastAPI()
model_db = {
    "gpt2": "checkpoints/gpt2-pytorch_model.bin",
}

seed = random.randint(0, 2147483647)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = get_encoder()
config = GPT2Config()
map_location = 'cpu' if not torch.cuda.is_available() else None
model: Any = GPT2LMHeadModel(config)

@app.get("/api/v1/model/{version:str}/predict/{prompt:str}")
def predict(
    version: Literal["gpt2"] = fastapi.Path(..., ),
    prompt: str = fastapi.Path(...),
    temperature: Optional[float] = fastapi.Query(0.7),
    top_k: Optional[int] = fastapi.Query(1),
):


    logger.info(f"Loading model: {version=}")

    model_path = model_db[version] 
    state_dict = torch.load(model_path, map_location=map_location)
    pytorch_model = load_weight(model, state_dict)
    pytorch_model.to(device)
    pytorch_model.eval()
    logger.info("Encode prompt.")
    context_tokens = enc.encode(prompt)
    logger.info("Generate output.")
    generated = 0
    
    with redis_lock():
        out = sample_sequence(
            model = pytorch_model, 
            length = config.n_ctx // 2,
            context=context_tokens,
            start_token = None,
            batch_size = 1,
            temperature=temperature,
            top_k = top_k,
            device = device
        )
    

    out = out[:, len(context_tokens):].tolist()
    text = enc.decode(out[0])
    return { "ouput": text }














