from kithara.model.maxtext.inference_engine import MaxEngine
from kithara import MaxTextModel
from kithara.model.maxtext.conversion_utils import MaxTextConversionMixin
from transformers import AutoTokenizer
import jax 

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokens = tokenizer("what is your name?")
tokens = tokens["input_ids"]
print("tokens", tokens)
decoded = tokenizer.decode(tokens)
print("decoded", decoded)
tokens = jax.numpy.array(tokens)


# Create model
model = MaxTextModel.from_preset("hf://google/gemma-2-2b", scan_layers=True)

# Create inference engine
config = model.maxtext_config
params = MaxTextConversionMixin.get_maxtext_params(model)

engine = MaxEngine(config)
engine.load_existing_params(config, params)

# Currently this is single-prompt prefill
prefill_result, first_token = engine.prefill(
    params=params, padded_tokens=tokens, true_length=len(tokens)
)

# Do I have to do this for every new prompt?
decode_state = engine.init_decode_state()
decode_state = engine.insert(prefill_result, decode_state, slot=0)

sampled_tokens_list = []

for i in range(100):
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    sampled_tokens_list.append(int(decode_state["tokens"][0][0]))
    print(i, "sampled_tokens", decode_state["tokens"])

print("sampled_tokens_list", sampled_tokens_list)
decoded = tokenizer.decode(sampled_tokens_list)
print("decoded", decoded)
