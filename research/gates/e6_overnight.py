"""Autonomous overnight experiment driver.

Runs every remaining heavy experiment end to end, staged, checkpointed and resumable.
Designed to survive anything: each stage is independent, failures are logged and
skipped, and re-running the script picks up exactly where it stopped.

Launched by Windows Task Scheduler (not by a shell), so it survives terminal close,
parent-process death, and logout.  A watchdog task restarts it if it dies.

Stages, cheapest and highest-value first:
   collect_pd      Python docs 3.8 vs 3.13, section-level  (the documentation regime)
   collect_wp      Wikipedia lead sections 2023 vs 2026    (the wiki regime)
   collect_ax      expanded arXiv abstracts v1 vs latest   (the independent-edit regime)
   embed_*         MiniLM forward pass, ~2 doc/s, resumable per corpus
   analyze_*       E6: Moran's I, variogram, probe-kNN, position-vs-magnitude
   retrieval_*     selective re-index bake-off vs random / oracle / change-log
   gate1_real      threshold transfer under volume shift, on REAL embeddings
   baselines       cluster-threshold refresh, MMD, Page-Hinkley
   power_floor     detection power near the floor (currently saturated at 1.000)
"""
import json, os, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

STATUS = os.path.join(HERE, "status.json")
LOG = os.path.join(HERE, "overnight.log")
PY = sys.executable


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_status():
    if os.path.exists(STATUS):
        try:
            return json.load(open(STATUS, encoding="utf-8"))
        except Exception:
            pass
    return {"done": [], "failed": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_status(s):
    json.dump(s, open(STATUS, "w", encoding="utf-8"), indent=1)


def run_py(script, *args, timeout=7200):
    """Run a helper script in a subprocess so one crash cannot take the driver down."""
    cmd = [PY, "-u", script] + [str(a) for a in args]
    log("  $ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=HERE)
    tail = (p.stdout or "")[-4000:]
    if tail.strip():
        for ln in tail.strip().split("\n")[-25:]:
            log("    | " + ln)
    if p.returncode != 0:
        log("    ! exit %d  stderr: %s" % (p.returncode, (p.stderr or "")[-1500:]))
    return p.returncode == 0


# --------------------------------------------------------------------------- stages

CORPORA = [
    # name, corpus json, embedding tag
    ("pd", "pd_corpus.json", "_pd"),
    ("wp", "wp_corpus.json", "_wp"),
    ("ax", "ax_corpus_big.json", "_axbig"),
]


def stage_collect_pd():
    return run_py("pd_fetch.py", 100000, timeout=3600)


def stage_collect_wp():
    if os.path.exists("wp_corpus.json"):
        return True
    for attempt in range(6):
        if run_py("wp_fetch2.py", 22, timeout=3600):
            if os.path.exists("wp_corpus.json"):
                return True
        log("    wp attempt %d incomplete, retrying" % (attempt + 1))
        time.sleep(60)
    return os.path.exists("wp_corpus.json")


def stage_collect_ax():
    return run_py("ax_fetch_big.py", timeout=3600)


def make_embed_stage(name, corpus, tag):
    def f():
        if not os.path.exists(corpus):
            log("    %s missing, skipping" % corpus)
            return False
        ok = True
        for field, cache in (("text0", "C0%s.npy" % tag), ("text1", "C1%s.npy" % tag)):
            for attempt in range(12):          # resumable: each call advances the cache
                if run_py("embed_one.py", corpus, field, cache, timeout=3000):
                    break
                log("    embed %s %s attempt %d incomplete" % (name, field, attempt + 1))
            else:
                ok = False
        return ok
    return f


def make_analyze_stage(name, corpus, tag):
    def f():
        return run_py("e6_generic.py", corpus, tag, name, timeout=5400)
    return f


def make_retrieval_stage(name, corpus, tag):
    def f():
        return run_py("e6b_generic.py", corpus, tag, name, timeout=5400)
    return f


def stage_significance():
    ok = True
    for nm, cp, tg in [("ax", "ax_corpus_big.json", "_axbig"), ("pd", "pd_corpus.json", "_pd"),
                       ("wp", "wp_corpus.json", "_wp"), ("axs", "ax_corpus.json", "_ax")]:
        if os.path.exists(cp) and os.path.exists("C0%s.npy" % tg):
            ok &= run_py("significance.py", cp, tg, nm, 30, timeout=5400)
    return ok


def stage_gate1():
    return run_py("gate1_real.py", timeout=5400)


def stage_baselines():
    return run_py("baselines.py", timeout=5400)


def stage_power():
    return run_py("power_floor.py", timeout=5400)


def build_plan():
    # embed each corpus as soon as it exists: embedding is the long pole (~2 doc/s),
    # so it must not wait behind collection of a later corpus.
    plan = [("collect_pd", stage_collect_pd),
            ("embed_pd", make_embed_stage("pd", "pd_corpus.json", "_pd")),
            ("collect_wp", stage_collect_wp),
            ("embed_wp", make_embed_stage("wp", "wp_corpus.json", "_wp")),
            ("collect_ax", stage_collect_ax),
            ("embed_ax", make_embed_stage("ax", "ax_corpus_big.json", "_axbig"))]
    for name, corpus, tag in CORPORA:
        plan.append(("analyze_" + name, make_analyze_stage(name, corpus, tag)))
    for name, corpus, tag in CORPORA:
        plan.append(("retrieval_" + name, make_retrieval_stage(name, corpus, tag)))
    plan += [("significance", stage_significance),
             ("gate1_real", stage_gate1),
             ("baselines", stage_baselines),
             ("power_floor", stage_power)]
    return plan



def acquire_lock():
    """Single instance only. Takes over a stale lock whose PID is gone."""
    lf = os.path.join(HERE, "overnight.lock")
    if os.path.exists(lf):
        try:
            pid = int(open(lf).read().strip())
        except Exception:
            pid = -1
        alive = False
        if pid > 0:
            try:
                out = subprocess.run(["wmic", "process", "where", "processid=%d" % pid,
                                      "get", "CommandLine"], capture_output=True, text=True)
                alive = "overnight.py" in (out.stdout or "")
            except Exception:
                alive = False
        if alive:
            log("another driver (pid %d) is running; exiting" % pid)
            return False
        log("stale lock from pid %s, taking over" % pid)
    open(lf, "w").write(str(os.getpid()))
    return True


def main():
    if not acquire_lock():
        return
    s = load_status()
    log("=" * 70)
    log("overnight driver starting; %d stages already done" % len(s["done"]))
    for name, fn in build_plan():
        if name in s["done"]:
            continue
        log("STAGE %s" % name)
        t0 = time.time()
        try:
            ok = fn()
        except Exception:
            ok = False
            log("    EXCEPTION\n" + traceback.format_exc()[-2000:])
        dt = time.time() - t0
        if ok:
            s["done"].append(name)
            s.pop("current", None)
            log("STAGE %s OK (%.0fs)" % (name, dt))
        else:
            s["failed"][name] = s["failed"].get(name, 0) + 1
            log("STAGE %s FAILED (attempt %d, %.0fs)" % (name, s["failed"][name], dt))
            # a stage that has failed 3 times is marked done so the run can proceed
            if s["failed"][name] >= 3:
                s["done"].append(name)
                log("STAGE %s giving up after 3 attempts, moving on" % name)
        save_status(s)
    s["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_status(s)
    log("ALL STAGES COMPLETE")


if __name__ == "__main__":
    main()
