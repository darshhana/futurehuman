import statistics as st
import torch


def pad_turns(case, target_tokens, tok):
    """Prepend neutral chatter until the prompt is roughly target_tokens long (context-length scaling test)."""
    import copy
    c = copy.deepcopy(case)
    filler = ["The quarterly numbers look fine to me.", "Traffic was heavy near the station today.", "I will bring snacks for everyone."]
    i = 0
    while len(tok(" ".join(t for _, t in c["turns"]))["input_ids"]) < target_tokens:
        c["turns"].insert(0, ["other" if i % 2 else "user", filler[i % 3] + f" ({i})"])
        i += 1
        if i > 600:
            break
    return c


def time_generations(gen, messages_list, max_new_tokens=200):
    lat, tps, nin = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for m in messages_list:
        r = gen(m, thinking=False, max_new_tokens=max_new_tokens)
        lat.append(r["latency_s"])
        tps.append(r["n_out"] / r["latency_s"] if r["latency_s"] else 0)
        nin.append(r["n_in"])
    q = lambda a, p: sorted(a)[min(len(a) - 1, int(p * len(a)))]
    return dict(n=len(lat), latency_p50=st.median(lat), latency_p95=q(lat, 0.95), tokens_per_s=st.mean(tps),
                mean_prompt_tokens=st.mean(nin), peak_vram_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
