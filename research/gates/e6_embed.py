import json, os, sys, time, numpy as np
from minilm_np import MiniLM
path, field, cache = sys.argv[1], sys.argv[2], sys.argv[3]
docs = json.load(open(path, encoding="utf-8"))
texts = [d[field] for d in docs]
done, out = 0, []
if os.path.exists(cache):
    E = np.load(cache); done = len(E)
    if done >= len(texts):
        print(cache, "already complete", E.shape); sys.exit(0)
    out = [E]
model = MiniLM()
t0 = time.time()
for s in range(done, len(texts), 128):
    out.append(model.embed(texts[s:s+128], batch=32).astype(np.float32))
    cur = np.vstack(out); np.save(cache, cur)
    el = time.time() - t0; rate = (len(cur)-done)/max(el,1e-9)
    print("%s %d/%d %.2f doc/s eta %.0fs" % (cache, len(cur), len(texts), rate,
          (len(texts)-len(cur))/max(rate,1e-9)), flush=True)
print(cache, "DONE", np.vstack(out).shape)
