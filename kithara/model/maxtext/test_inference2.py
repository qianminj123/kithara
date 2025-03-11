from kithara.model.maxtext.inference_engine import MaxEngine
from kithara import MaxTextModel
from kithara.model.maxtext.conversion_utils import MaxTextConversionMixin
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
import numpy as np

def batch_decode(model, engine, tokenizer, prompts, max_new_tokens=100):
    """
    Perform batch decoding with MaxEngine.
    
    Args:
        model: MaxTextModel instance
        engine: MaxEngine instance
        tokenizer: Tokenizer instance
        prompts: List of strings to decode
        max_new_tokens: Maximum number of tokens to generate per prompt
        
    Returns:
        List of decoded responses
    """
    # Get params
    params = MaxTextConversionMixin.get_maxtext_params(model)
    
    # Tokenize all prompts
    tokenized_prompts = [tokenizer(prompt, return_tensors="np")["input_ids"][0] for prompt in prompts]
    
    # Calculate the total sequence length and padding
    max_len = max(len(tokens) for tokens in tokenized_prompts)
    
    # Create padded tokens, positions and segment IDs for prefill_concat
    batch_size = len(prompts)
    padded_tokens = np.zeros((max_len * batch_size), dtype=np.int32)
    decoder_positions = np.zeros((max_len * batch_size), dtype=np.int32)
    decoder_segment_ids = np.zeros((max_len * batch_size), dtype=np.int32)
    
    # Start positions and true lengths for each prompt
    start_pos = np.zeros(batch_size, dtype=np.int32)
    true_lengths = np.zeros(batch_size, dtype=np.int32)
    
    # Populate the arrays
    current_pos = 0
    for i, tokens in enumerate(tokenized_prompts):
        length = len(tokens)
        true_lengths[i] = length
        start_pos[i] = current_pos
        
        # Fill padded_tokens with actual tokens
        padded_tokens[current_pos:current_pos+length] = tokens
        
        # Fill position indices (0 to length-1 for each prompt)
        decoder_positions[current_pos:current_pos+length] = np.arange(length)
        
        # Mark segment IDs with the prompt index to distinguish between prompts
        decoder_segment_ids[current_pos:current_pos+length] = i + 1  # Using i+1 to distinguish from padding (0)
        
        current_pos += length
    
    # Convert to JAX arrays
    padded_tokens = jnp.array(padded_tokens)
    decoder_positions = jnp.array(decoder_positions)
    decoder_segment_ids = jnp.array(decoder_segment_ids)
    start_pos = jnp.array(start_pos)
    true_lengths = jnp.array(true_lengths)
    
    # Initialize the decode state
    decode_state = engine.init_decode_state()
    
    print("padded_tokens", padded_tokens)
    print("decoder_positions", decoder_positions)
    print("decoder_segment_ids", decoder_segment_ids)
    print("start_pos", start_pos)
    print("true_lengths", true_lengths)
    print("num_prompts", batch_size)
    # Call prefill_concat
    cache, prefill_results, first_tokens = engine.prefill_concat(
        params=params,
        padded_tokens=padded_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        start_pos=start_pos,
        true_lengths=true_lengths,
        num_prompts=batch_size,
        rng=jax.random.PRNGKey(0)
    )
    print("prefill_results", prefill_results)
    print("first_tokens", first_tokens)
    # Assign slots for each prompt
    
    for i in range(batch_size):    
        # Insert the prefilled results into the decode state
        decode_state = engine.insert_partial(
            prefix=prefill_results[i],
            decode_state=decode_state[i],
            slots=i,
        )
        
    
    # Store generated tokens for each prompt
    generated_tokens = [[] for _ in range(batch_size)]
    
    # Add first tokens from prefill
    for i in range(batch_size):
        token_val = int(decode_state["tokens"][i][0])
        generated_tokens[i].append(token_val)
    
    # Generate additional tokens
    for step in range(max_new_tokens - 1):  # -1 because we already have the first token
        decode_state, _ = engine.generate(params, decode_state)
        
        # Collect generated tokens for each prompt
        for i in range(batch_size):
            token_val = int(decode_state["tokens"][i][0])
            generated_tokens[i].append(token_val)
            
            # Optional: Check for end of sequence tokens and stop early
            # if token_val == tokenizer.eos_token_id:
            #     break
    
    # Decode the generated tokens
    decoded_outputs = [tokenizer.decode(tokens) for tokens in generated_tokens]
    
    return decoded_outputs


# Example usage:
if __name__ == "__main__":
    # Create tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = MaxTextModel.from_preset("hf://google/gemma-2-2b", scan_layers=True)
    
    # Create inference engine
    config = model.maxtext_config
    params = MaxTextConversionMixin.get_maxtext_params(model)
    engine = MaxEngine(config)
    engine.load_existing_params(params)
    
    # Define prompts
    prompts = [
        "What is artificial intelligence?",
        "Tell me about quantum computing",
        "Explain machine learning in simple terms"
    ]
    
    # Batch decode and get results
    results = batch_decode(model, engine, tokenizer, prompts, max_new_tokens=50)
    
    # Print results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response: {result}")
        print("=" * 50)