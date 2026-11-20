import gc, time


def load_qwen(model_id="Qwen/Qwen3-4B", quant="4bit", adapter=None):
    """quant: '4bit' | '8bit' | 'fp16'. No trust_remote_code needed for Qwen3."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(device_map="auto")
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    elif quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kw)
    if adapter:  # LoRA/QLoRA adapter from the training stage: same harness, same tests
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def free(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def make_gen(model, tok, seed=0):
    """Returns gen(messages, thinking=False, max_new_tokens=200) -> dict(text, latency_s, n_in, n_out).
    thinking=False -> greedy (reproducible).  thinking=True -> Qwen's recommended sampling (greedy loops in think mode)."""
    import torch

    def gen(messages, thinking=False, max_new_tokens=200):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        if thinking:
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        else:
            kw.update(do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, **kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0][inputs["input_ids"].shape[1]:]
        return dict(text=tok.decode(new, skip_special_tokens=True), latency_s=dt,
                    n_in=int(inputs["input_ids"].shape[1]), n_out=int(len(new)))
    return gen
