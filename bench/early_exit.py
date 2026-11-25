# Early-exit self-drafting: how often does the model after 1 or 2 of its 3 loop passes predict what the full
# model predicts? Teacher-forced on traced opencode generations.
import ctypes, json, sys, time
import numpy as np
lib = ctypes.CDLL(sys.argv[1]); model = sys.argv[2]; trace = sys.argv[3]; nreq = int(sys.argv[4]); maxgen = int(sys.argv[5])
f32p = ctypes.POINTER(ctypes.c_float)
lib.ink_open.restype = ctypes.c_void_p; lib.ink_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ink_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, f32p]
lib.ink_step.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int, f32p]
lib.ink_draft.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int, ctypes.c_int, f32p]
lib.ink_n_vocab.argtypes = [ctypes.c_void_p]; lib.ink_error.restype = ctypes.c_char_p
h = lib.ink_open(model.encode(), 40960, 0, 4)
if not h: sys.exit(lib.ink_error().decode())
V = lib.ink_n_vocab(h)
bufs = {k: np.zeros(V, np.float32) for k in ("full", 1, 2, 3)}
ptr = {k: v.ctypes.data_as(f32p) for k, v in bufs.items()}
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()
rows = [json.loads(l) for l in open(trace)]
rows = [r for r in rows if len(r["prompt"]) < 36000 and len(r["gen"]) > 150][:nreq]
stats = {1: [0, 0.0], 2: [0, 0.0]}; n = 0; chk = []
for r in rows:
    p = r["prompt"]; arr = (ctypes.c_int32 * len(p))(*p)
    assert lib.ink_prefill(h, arr, len(p), 0, None) == 0
    g = r["gen"][:maxgen]; t0 = time.time()
    for i, t in enumerate(g[:-1]):
        pos = len(p) + i
        for k in (1, 2): assert lib.ink_draft(h, t, pos, k, ptr[k]) == 0
        if len(chk) < 6: assert lib.ink_draft(h, t, pos, 3, ptr[3]) == 0
        assert lib.ink_step(h, t, pos, ptr["full"]) == 0
        pf = softmax(bufs["full"].astype(np.float64)); top = int(bufs["full"].argmax())
        if len(chk) < 6: chk.append((int(bufs[3].argmax()) == top, float(np.abs(bufs[3] - bufs["full"]).max())))
        for k in (1, 2):
            d = int(bufs[k].argmax())
            stats[k][0] += d == top; stats[k][1] += pf[d]
        n += 1
    print(f"request: prompt {len(p)}, {len(g)} tokens in {time.time()-t0:.0f} s; running: " +
          ", ".join(f"{k} pass{'es' if k > 1 else ''}: top-1 {stats[k][0]/n:.0%}, acceptance {stats[k][1]/n:.0%}" for k in (1, 2)), flush=True)
print("check, 3-pass draft vs full step (same top, max |logit diff|):", chk)
