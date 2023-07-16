
import fastapi
import torch
import uvicorn
import torch


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


app = fastapi.FastAPI()

model_version_to_path = {
    "gpt2": "checkpoints/gpt2-pytorch_model.bin"
}



seed = random.randint(0, 2147483647)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = get_encoder()
config = GPT2Config()
map_location = 'cpu' if not torch.cuda.is_available() else None


class Args(BaseModel):
    text: str
    quiet: Optional[bool] = False
    nsamples: int = 1
    unconditional: Optional[bool] = False
    batch_size: int = 1
    length: Optional[int] = -1
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 1
    model: Any = GPT2LMHeadModel(config)







@app.get("/api/v1/model/{version:str}/predict/{prompt:str}")
def predict(
    version: Literal["gpt2"] = fastapi.Path(..., ),
    prompt: str = fastapi.Path(...)
):
    global model
    args = Args(text=prompt)
    path_to_model = model_version_to_path[version]

    state_dict = torch.load(
        'checkpoints/gpt2-pytorch_model.bin',
        map_location=map_location
    )

    assert args.nsamples % args.batch_size == 0
    

    pytorch_model = load_weight(args.model, state_dict)
    pytorch_model.to(device)
    pytorch_model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    context_tokens = enc.encode(args.text)
    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=pytorch_model, 
            length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
    
    return { "ouput": text }














