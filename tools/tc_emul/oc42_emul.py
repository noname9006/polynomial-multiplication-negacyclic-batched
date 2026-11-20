# OC42 emulator: full NTT N=65536, one prime, plain modular arithmetic.
# Mirrors r8f_emul.py but encodes only stages 0..4 in encode256, then fused_high
# performs forward stages 5,6,7,8 by lane-xor shfl (low bit m == lane bit m-4),
# followed by the unchanged high stages / pointwise / inverse.
# Reference: plain 16-stage merged-psi forward DIT * pointwise * inverse DIF.
# Negative test: corrupt one twiddle -> must produce mismatches.
import sys
import numpy as np

n = 16
N = 1 << n
p = 2013265921
g = 31
psi = pow(g, (p - 1) // (2 * N), p)
assert pow(psi, N, p) == p - 1
tab = [1] * (2 * N)
for e in range(1, 2 * N):
    tab[e] = tab[e - 1] * psi % p
T = np.array(tab, dtype=np.int64)
TI = np.array([tab[(2 * N - e) % (2 * N)] for e in range(2 * N)], dtype=np.int64)


def mm(a, b):
    return (a.astype(object) * b.astype(object) % p).astype(np.int64)


def fw(m, j):
    return T[(1 << (n - 1 - m)) * (1 + 2 * j)]


def iw(m, j):
    return TI[(1 << (n - 1 - m)) * (1 + 2 * j)]


def dit_stage(X, m):
    half = 1 << m
    x = np.arange(N)
    lo = x[(x & half) == 0]
    j = lo & (half - 1)
    u, v = X[lo], mm(X[lo + half], fw(m, j))
    Y = X.copy()
    Y[lo], Y[lo + half] = (u + v) % p, (u - v) % p
    return Y


def dif_stage(X, m):
    half = 1 << m
    x = np.arange(N)
    lo = x[(x & half) == 0]
    j = lo & (half - 1)
    u, v = X[lo], X[lo + half]
    Y = X.copy()
    Y[lo], Y[lo + half] = (u + v) % p, mm((u - v) % p, iw(m, j))
    return Y


def run(oc42, corrupt=-1):
    rng = np.random.default_rng(1)
    A = rng.integers(0, p, N).astype(np.int64)
    B = rng.integers(0, p, N).astype(np.int64)

    # reference
    RA, RB = A.copy(), B.copy()
    for m in range(16):
        RA, RB = dit_stage(RA, m), dit_stage(RB, m)
    RC = mm(RA, RB)
    for m in range(15, -1, -1):
        RC = dif_stage(RC, m)

    # encode256: stages 0..4 under OC42, 0..7 otherwise
    EA, EB = A.copy(), B.copy()
    nst = 6 if oc42 else 8
    for m in range(nst):
        EA, EB = dit_stage(EA, m), dit_stage(EB, m)
    ra = np.full(N, -1, dtype=np.int64)
    rb = np.full(N, -1, dtype=np.int64)
    for start in range(0, N, 256):
        row, hi = start >> 9, (start >> 8) & 1
        for r in range(256):
            bxx, l4 = (r >> 5) + ((r >> 4) & 1) * 8, r & 15
            pos = ((l4 >> 2) << 6) | (((l4 >> 1) & 1) << 5) | (bxx << 1) | (l4 & 1)
            ad = (bxx * (N >> 9) + row) * 32 + (hi << 4) + l4
            assert ra[ad] == -1
            ra[ad] = EA[start + pos]
            rb[ad] = EB[start + pos]
    assert (ra >= 0).all()

    # fused_high(r8f)
    HC = 128
    out = np.full(N, -1, dtype=np.int64)
    lane = np.arange(32)
    for bx in range(16):
        low = ((lane >> 2) << 6) | (((lane >> 1) & 1) << 5) | (bx << 1) | (lane & 1)
        rows = np.arange(HC)
        ad = (bx * HC + rows[:, None]) * 32 + lane[None, :]
        av, bv = ra[ad].copy(), rb[ad].copy()
        X = rows[:, None] * 512 + low[None, :]

        if oc42:
            # forward stages 6,7 by lane^4,^8 (no transpose)
            for m in (6, 7):
                lb = 1 << (m - 4)
                wf5 = fw(m, low & ((1 << m) - 1)).astype(object)
                if m == corrupt:
                    wf5 = (wf5 + 1) % p
                xa = av[:, lane ^ lb]
                xb = bv[:, lane ^ lb]
                up = (lane & lb) != 0
                nav = np.where(up[None, :], (xa - mm(av, wf5[None, :])) % p,
                               (av + mm(xa, wf5[None, :])) % p)
                nbv = np.where(up[None, :], (xb - mm(bv, wf5[None, :])) % p,
                               (bv + mm(xb, wf5[None, :])) % p)
                av, bv = nav, nbv

        # stage 8 (base already does this; use plain shfl form)
        w8 = fw(8, low & 255).astype(object)
        if corrupt == 8:
            w8 = (w8 + 1) % p
        xa = av[:, lane ^ 16]
        xb = bv[:, lane ^ 16]
        up = (lane & 16) != 0
        av = np.where(up[None, :], (xa - mm(av, w8[None, :])) % p,
                      (av + mm(xa, w8[None, :])) % p)
        bv = np.where(up[None, :], (xb - mm(bv, w8[None, :])) % p,
                      (bv + mm(xb, w8[None, :])) % p)

        # high stages 9..15
        for st in range(7):
            m = 9 + st
            H = 1 << st
            for V in (av, bv):
                for h in range(HC):
                    if h & H:
                        continue
                    j = X[h] & ((1 << m) - 1)
                    u, v = V[h].copy(), mm(V[h + H], fw(m, j))
                    V[h], V[h + H] = (u + v) % p, (u - v) % p

        cv = mm(av, bv)
        for st in range(6, -1, -1):
            m = 9 + st
            H = 1 << st
            for h in range(HC):
                if h & H:
                    continue
                j = X[h] & ((1 << m) - 1)
                u, v = cv[h].copy(), cv[h + H].copy()
                cv[h], cv[h + H] = (u + v) % p, mm((u - v) % p, iw(m, j))

        # inverse low stages 8..5 by shfl
        for m in range(8, 4, -1):
            half, lb = 1 << m, 1 << (m - 4)
            w = iw(m, low & (half - 1))
            hi = (lane & lb) != 0
            x = cv[:, lane ^ lb]
            cv = np.where(hi[None, :], mm((x - cv) % p, w[None, :]), (cv + x) % p)

        inner = (bx << 1) | (lane & 1)
        xx = X
        oad = ((xx >> 5) * 1 + 0) * 32 + inner[None, :]
        assert (out[oad] == -1).all()
        out[oad] = cv
    assert (out >= 0).all()

    # crt_tc2(r8f): 32-run, thread=limb DIF 4..0
    res = np.zeros(N, dtype=np.int64)
    for blk in range(N // 32):
        v = [int(t) for t in out[blk * 32:(blk + 1) * 32]]
        tw = [int(iw(m, jj)) for m in range(5) for jj in range(1 << m)]
        for m in range(4, -1, -1):
            half = 1 << m
            for i in range(32):
                if i & half:
                    continue
                u, x = v[i], v[i + half]
                v[i] = (u + x) % p
                v[i + half] = (u - x) * tw[(half - 1) + (i & (half - 1))] % p
        res[blk * 32:(blk + 1) * 32] = v
    return int((res != RC).sum()), N


if __name__ == "__main__":
    bad, N = run(oc42=True)
    print("OC42 full NTT: mismatches =", bad, "of", N)
    if bad:
        sys.exit(1)
    bad0, _ = run(oc42=False)
    print("base path   : mismatches =", bad0, "of", N)
    if bad0:
        sys.exit(1)
    # negative test: corrupt forward stage 6 twiddle
    bneg, _ = run(oc42=True, corrupt=6)
    print("negative (corrupt twiddle m=6): mismatches =", bneg)
    if bneg == 0:
        print("FAIL: negative test did not detect corruption")
        sys.exit(1)
    print("OC42 emulation OK")
