"""Sweep cache_bias values to find optimal routing bias."""
import json, os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "benchmarks/cache_bias_sweep_20260331.txt"
JSON_OUT = "benchmarks/cache_bias_sweep_20260331.json"

BIAS_CODE = '''
import json, time, torch, sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from tinyserve.offload import load_and_offload
from scripts.prompts import COLD_START, CODE_PROMPTS, CREATIVE_PROMPTS, MULTILINGUAL_PROMPTS

BIAS = {bias}
tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = load_and_offload("openai/gpt-oss-20b", cache_bias=BIAS, attn_implementation="sdpa")
cache = next(p.cache for p in model._offload_pipelines if p.cache is not None)

prompts = COLD_START[:4] + CODE_PROMPTS[:2] + CREATIVE_PROMPTS[:2] + MULTILINGUAL_PROMPTS[:2]
inp = tok("Hello", return_tensors="pt").to("cuda")
with torch.inference_mode():
    model.generate(**inp, max_new_tokens=3, do_sample=False)

cache.reset_stats()
total_tok, t0 = 0, time.perf_counter()
for prompt in prompts:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=25, do_sample=False)
    total_tok += out.shape[1] - inp["input_ids"].shape[1]

elapsed = time.perf_counter() - t0
hr = cache.hits / (cache.hits + cache.misses) if (cache.hits + cache.misses) > 0 else 0
tps = total_tok / elapsed
layer_stats = cache.get_layer_stats()
deep_hr = sum(layer_stats.get(li, {{}}).get("hit_rate", 0) for li in range(18, 24)) / 6

print(json.dumps({{"bias": BIAS, "hr": round(hr, 4), "tps": round(tps, 1), "deep_hr": round(deep_hr, 4)}}))
'''

def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

if __name__ == "__main__":
    with open(LOG, "w") as f: f.write("=== Cache Bias Sweep ===\n\n")

    biases = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = {}

    log(f"{'bias':>6} {'HR%':>6} {'tok/s':>7} {'Deep(18-23)':>12}")
    log("-" * 35)

    for bias in biases:
        code = BIAS_CODE.format(bias=bias)
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env)
        if r.returncode == 0:
            try:
                data = json.loads(r.stdout)
                log(f"{bias:>6.1f} {data['hr']:>5.1%} {data['tps']:>6.1f} {data['deep_hr']:>10.0%}")
                results[str(bias)] = data
            except json.JSONDecodeError:
                log(f"{bias:>6.1f}  ERR: bad JSON")
        else:
            log(f"{bias:>6.1f}  ERR: {r.stderr[-100:]}")

    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved to {JSON_OUT}")
