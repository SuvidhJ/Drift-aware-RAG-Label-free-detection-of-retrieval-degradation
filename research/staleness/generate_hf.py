"""Run a local instruct model over prompts.jsonl -> outputs.jsonl.

Standalone on purpose: it needs torch and transformers, which the project venv does not
carry. Greedy decoding, so the run is deterministic and re-runnable.

    <venv with torch> python research/staleness/generate_hf.py [model_dir]

Resumable: outputs already written are skipped.
"""
import json
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parents[2] / "embeddings" / "cache" / "staleness" / "generation"
MODEL = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/.cache/qwen2.5-0.5b-instruct").expanduser()
PROMPTS = sys.argv[2] if len(sys.argv) > 2 else "prompts.jsonl"
OUTPUTS = sys.argv[3] if len(sys.argv) > 3 else "outputs.jsonl"

SYSTEM = "You answer strictly from the provided context and never from memory."
TEMPLATE = (
    "Context:\n{context}\n\n"
    "Fill in each blank (___) in the statement below with the exact number taken from "
    "the context. Reply with only the number(s), separated by commas. If the context "
    "does not contain it, reply: unknown\n\nStatement: {statement}"
)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
torch.set_num_threads(max(torch.get_num_threads(), 4))

items = [json.loads(l) for l in open(HERE / PROMPTS)]
out_path = HERE / OUTPUTS
done = sum(1 for _ in open(out_path)) if out_path.exists() else 0
print(f"{len(items)} prompts, {done} already done, model {MODEL.name}", flush=True)

t0 = time.time()
with open(out_path, "a") as f:
    for n, it in enumerate(items[done:], start=done):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEMPLATE.format(**it)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)
        ids = enc["input_ids"]
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=enc["attention_mask"], max_new_tokens=24,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        it["output"] = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
        it["model"] = MODEL.name
        f.write(json.dumps(it) + "\n")
        f.flush()
        if (n + 1) % 50 == 0:
            rate = (n + 1 - done) / (time.time() - t0)
            print(f"  {n+1}/{len(items)}  {rate:.2f}/s  eta {(len(items)-n-1)/rate/60:.0f} min",
                  flush=True)
print("GENERATION DONE", flush=True)
