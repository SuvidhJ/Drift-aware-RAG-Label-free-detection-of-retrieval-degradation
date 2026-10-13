import json
from vcommon import P9, gen_cache, gkey, sample
_, _, qs = sample()
ctx = json.load(open("ctx.json"))
for m in ("qwen2.5:14b-instruct",):
    c = gen_cache(m)
    used = set()
    for k, texts in ctx.items():
        retr, pol, n = k.split("|"); n = int(n)
        cx = "\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(texts))
        used.add(gkey(m, P9.TEMPLATE.format(context=cx, statement=qs[n]["statement"])))
    hit = used & set(c)
    print(m, "cache entries", len(c), "| keys of final-run scale prompts present", len(hit), "| cache entries not from final scale prompts", len(set(c) - used))
