import json, os, subprocess, sys, time, numpy as np
from minilm_np import MiniLM

path, field, cache = sys.argv[1], sys.argv[2], sys.argv[3]
LOCK = cache + ".lock"


def pid_alive(pid):
    try:
        out = subprocess.run(["wmic", "process", "where", "processid=%d" % pid,
                              "get", "CommandLine"], capture_output=True, text=True)
        return "embed_one.py" in (out.stdout or "")
    except Exception:
        return False


def acquire(max_wait=1500):
    """Only one process may append to a given cache. Wait out a live holder."""
    waited = 0
    while True:
        if os.path.exists(LOCK):
            try:
                pid = int(open(LOCK).read().strip())
            except Exception:
                pid = -1
            if pid != os.getpid() and pid > 0 and pid_alive(pid):
                if waited >= max_wait:
                    print("%s locked by pid %d, giving up" % (cache, pid))
                    return False
                time.sleep(20)
                waited += 20
                continue
        open(LOCK, "w").write(str(os.getpid()))
        return True


if not acquire():
    sys.exit(3)
try:
    docs = json.load(open(path, encoding="utf-8"))
    texts = [d[field] for d in docs]
    done, out = 0, []
    if os.path.exists(cache):
        E = np.load(cache)
        done = len(E)
        if done >= len(texts):
            print(cache, "already complete", E.shape)
            raise SystemExit(0)
        out = [E]
    model = MiniLM()
    t0 = time.time()
    for s in range(done, len(texts), 128):
        out.append(model.embed(texts[s:s + 128], batch=32).astype(np.float32))
        cur = np.vstack(out)
        np.save(cache, cur)
        open(LOCK, "w").write(str(os.getpid()))          # refresh
        el = time.time() - t0
        rate = (len(cur) - done) / max(el, 1e-9)
        print("%s %d/%d %.2f doc/s eta %.0fs" % (cache, len(cur), len(texts), rate,
              (len(texts) - len(cur)) / max(rate, 1e-9)), flush=True)
    print(cache, "DONE", np.vstack(out).shape)
finally:
    try:
        if os.path.exists(LOCK) and int(open(LOCK).read().strip()) == os.getpid():
            os.remove(LOCK)
    except Exception:
        pass
