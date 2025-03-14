from kithara.model.maxtext.inference_engine import MaxtextInferenceEngine as MaxEngine
from kithara import MaxTextModel
from kithara.model.maxtext.conversion_utils import MaxTextConversionMixin
from transformers import AutoTokenizer
import jax


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

def get_input(str_input):
    tokens = tokenizer(str_input)
    tokens = jax.numpy.array(tokens["input_ids"])
    return tokens

# Create model
model = MaxTextModel.from_random("gemma2-2b", seq_len=512)

inputs = ["what is your name?", "what is 1+1?", "how is an apple so round?", "what is the meaning of life?", "where is Peru?", "is the sky blue?"]

# predictions = model.generate(inputs, tokenizer=tokenizer)
# print("predictions", predictions)

import numpy as np 

inputs = ["what is your name?", "what is your name?", "what is your name?", "what is your name?"]

predictions = model.generate(inputs, tokenizer=tokenizer, max_length=1530, return_decoded=False)
print("predictions", predictions)
print("len", len(predictions), len(predictions[0]))

