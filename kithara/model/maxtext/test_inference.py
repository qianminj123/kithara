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
model = MaxTextModel.from_preset("hf://google/gemma-2-2b")

# Create inference engine
config = model.maxtext_config
params = MaxTextConversionMixin.get_maxtext_params(model)

engine = MaxEngine(config)
engine.load_existing_params(params)

batch_size = engine.max_concurrent_decodes

inputs = ["what is your name?", "what is 1+1?", "how is an apple so round?", "what is the meaning of life?", "where is Peru?", "is the sky blue?"]

decode_state = engine.init_decode_state()

for slot, str_input in enumerate(inputs):
    tokens = get_input(str_input)
    prefill_result, first_token = engine.prefill(
        params=params, padded_tokens=tokens, true_length=len(tokens)
    )
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)


sampled_tokens_list = [[] for _ in inputs]

for i in range(100):
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    for sample_idx in range(len(inputs)):
        sampled_tokens_list[sample_idx].append(decode_state["tokens"][sample_idx][0])

print("there are this many samples:", len(sampled_tokens_list))
for sample in sampled_tokens_list:
    decoded = tokenizer.decode(sample)
    print("decoded:", decoded)
