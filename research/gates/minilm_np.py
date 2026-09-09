"""all-MiniLM-L6-v2 forward pass in pure numpy, from the safetensors checkpoint.

No torch / transformers.  Implements BertTokenizer(do_lower_case=True) + 6-layer BERT
+ mean pooling over the attention mask + L2 normalise, matching modules.json.
"""
import json, struct, unicodedata, pathlib, numpy as np

ROOT = pathlib.Path.home()/".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# ---------------------------------------------------------------- weights
def load_safetensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        blob = fh.read()
    dt = {"F32": np.float32, "I64": np.int64, "F16": np.float16}
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__": continue
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(blob[s:e], dtype=dt[v["dtype"]]).reshape(v["shape"])
    return out

# ---------------------------------------------------------------- tokenizer
class Tok:
    def __init__(self, vocab_file):
        self.vocab = {}
        with open(vocab_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.unk, self.cls, self.sep, self.pad = (
            self.vocab["[UNK]"], self.vocab["[CLS]"], self.vocab["[SEP]"], self.vocab["[PAD]"])

    @staticmethod
    def _basic(text):
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or unicodedata.category(ch) == "Cc":
                out.append(" ")
            elif (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126) \
                 or unicodedata.category(ch).startswith("P"):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        return "".join(out).split()

    def _wordpiece(self, word):
        if len(word) > 100: return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end]
                if start > 0: sub = "##" + sub
                if sub in self.vocab: cur = self.vocab[sub]; break
                end -= 1
            if cur is None: return [self.unk]
            ids.append(cur); start = end
        return ids

    def encode(self, text, max_len=256):
        ids = [self.cls]
        for w in self._basic(text):
            ids += self._wordpiece(w)
            if len(ids) >= max_len - 1: break
        ids = ids[:max_len - 1] + [self.sep]
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        ids = ids + [self.pad] * (max_len - len(ids))
        return ids, mask

# ---------------------------------------------------------------- model
def layer_norm(x, w, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b

def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

class MiniLM:
    def __init__(self, root=ROOT, max_len=256):
        self.W = load_safetensors(pathlib.Path(root)/"model.safetensors")
        self.tok = Tok(pathlib.Path(root)/"vocab.txt")
        self.max_len, self.H, self.nh, self.L = max_len, 384, 12, 6
        self.dh = self.H // self.nh

    def _encode_batch(self, ids, mask):
        W = self.W
        B, T = ids.shape
        x = (W["embeddings.word_embeddings.weight"][ids]
             + W["embeddings.position_embeddings.weight"][None, :T, :]
             + W["embeddings.token_type_embeddings.weight"][0][None, None, :])
        x = layer_norm(x, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"])
        add = (1.0 - mask[:, None, None, :].astype(np.float32)) * -1e4

        for l in range(self.L):
            p = f"encoder.layer.{l}."
            def lin(h, name):
                return h @ W[p + name + ".weight"].T + W[p + name + ".bias"]
            q = lin(x, "attention.self.query").reshape(B, T, self.nh, self.dh).transpose(0, 2, 1, 3)
            k = lin(x, "attention.self.key").reshape(B, T, self.nh, self.dh).transpose(0, 2, 1, 3)
            v = lin(x, "attention.self.value").reshape(B, T, self.nh, self.dh).transpose(0, 2, 1, 3)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dh) + add
            s = s - s.max(-1, keepdims=True)
            a = np.exp(s); a /= a.sum(-1, keepdims=True)
            c = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, self.H)
            c = lin(c, "attention.output.dense")
            x = layer_norm(x + c, W[p + "attention.output.LayerNorm.weight"],
                                  W[p + "attention.output.LayerNorm.bias"])
            h = gelu(lin(x, "intermediate.dense"))
            h = lin(h, "output.dense")
            x = layer_norm(x + h, W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"])

        m = mask[:, :, None].astype(np.float32)
        emb = (x * m).sum(1) / np.maximum(m.sum(1), 1e-9)          # mean pooling
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def embed(self, texts, batch=32, verbose=False):
        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = [self.tok.encode(t, self.max_len) for t in chunk]
            ids = np.array([e[0] for e in enc], dtype=np.int64)
            msk = np.array([e[1] for e in enc], dtype=np.int64)
            T = max(int(msk.sum(1).max()), 1)
            out.append(self._encode_batch(ids[:, :T], msk[:, :T]).astype(np.float32))
            if verbose and (i // batch) % 20 == 0:
                print(f"    embedded {i + len(chunk)}/{len(texts)}", flush=True)
        return np.vstack(out)

if __name__ == "__main__":
    m = MiniLM()
    s = ["The cat sits on the mat.", "A feline rests upon the rug.",
         "Quantum chromodynamics describes the strong interaction.",
         "The Reserve Bank of India sets monetary policy."]
    e = m.embed(s)
    print("shape:", e.shape, " norms:", np.round(np.linalg.norm(e, axis=1), 5))
    C = e @ e.T
    print("cos matrix:\n", np.round(C, 3))
    print("\nSANITY: paraphrase(0,1)=%.3f should be >> unrelated(0,2)=%.3f" % (C[0,1], C[0,2]))
