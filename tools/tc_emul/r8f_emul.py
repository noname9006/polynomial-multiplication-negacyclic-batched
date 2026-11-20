# R8F layout/stage emulator (N=65536, one prime, plain modular arithmetic).
# Checks that encode256(r8f) store -> fused_high(r8f) lanes/shfl stages/output
# -> crt_tc2(r8f) read + shfl stages reproduce the plain 16-stage merged-psi
# forward DIT * pointwise * inverse DIF result slot by slot.
import numpy as np

n = 16
N = 1 << n
p = 2013265921  # 15*2^27+1, has 2N-th roots
g = 31
psi = pow(g, (p - 1) // (2 * N), p)
assert pow(psi, N, p) == p - 1
ipsi = pow(psi, p - 2, p)
PW = np.array([pow(psi, e, p) for e in range(1)] , dtype=object)
# psi^e table for e < 2N
tab = [1] * (2 * N)
for e in range(1, 2 * N):
    tab[e] = tab[e - 1] * psi % p
T = np.array(tab, dtype=np.int64)
TI = np.array([tab[(2 * N - e) % (2 * N)] for e in range(2 * N)], dtype=np.int64)


def mm(a, b):
    return (a.astype(object) * b.astype(object) % p).astype(np.int64)


def fw(m, j):  # forward twiddle of stage m (half 2^m), j < half
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


rng = np.random.default_rng(1)
A = rng.integers(0, p, N).astype(np.int64)
B = rng.integers(0, p, N).astype(np.int64)

# reference (all stages on the plain slot array)
RA, RB = A.copy(), B.copy()
for m in range(16):
    RA, RB = dit_stage(RA, m), dit_stage(RB, m)
RC = mm(RA, RB)
for m in range(15, -1, -1):
    RC = dif_stage(RC, m)

# --- encode256: stages 0..7 (tile-local), then the r8f store loop
EA, EB = A.copy(), B.copy()
for m in range(8):
    EA, EB = dit_stage(EA, m), dit_stage(EB, m)
ra = np.full(N, -1, dtype=np.int64)
rb = np.full(N, -1, dtype=np.int64)
for start in range(0, N, 256):
    row, hi = start >> 9, (start >> 8) & 1
    for r in range(256):
        bxx, l4 = r >> 4, r & 15
        pos = ((l4 >> 2) << 6) | (((l4 >> 1) & 1) << 5) | (bxx << 1) | (l4 & 1)
        ad = (bxx * (N >> 9) + row) * 32 + (hi << 4) + l4
        assert ra[ad] == -1
        ra[ad] = EA[start + pos]
        rb[ad] = EB[start + pos]
assert (ra >= 0).all()

# --- fused_high (r8f): per CTA bx, V[row][lane]
HC = 128
out = np.full(N, -1, dtype=np.int64)
lane = np.arange(32)
for bx in range(16):
    low = ((lane >> 2) << 6) | (((lane >> 1) & 1) << 5) | (bx << 1) | (lane & 1)
    rows = np.arange(HC)
    ad = (bx * HC + rows[:, None]) * 32 + lane[None, :]
    av, bv = ra[ad].copy(), rb[ad].copy()
    X = rows[:, None] * 512 + low[None, :]  # slot of each element
    # stage 8 by shfl lane^16
    w8 = fw(8, low & 255)
    for V in (av, bv):
        xa = V[:, lane ^ 16]
        up = (lane & 16) != 0
        new = np.where(up[None, :], (xa - mm(V, w8[None, :])) % p, (V + mm(xa, w8[None, :])) % p)
        V[:] = new
    # high stages 9..15 across rows (global-formula twiddles)
    for st in range(7):
        m = 9 + st
        H = 1 << st
        for V in (av, bv):
            for h in range(HC):
                if h & H:
                    continue
                j = (X[h] & ((1 << m) - 1))
                u, v = V[h].copy(), mm(V[h + H], fw(m, j))
                V[h], V[h + H] = (u + v) % p, (u - v) % p
    cv = mm(av, bv)
    for st in range(6, -1, -1):
        m = 9 + st
        H = 1 << st
        for h in range(HC):
            if h & H:
                continue
            j = (X[h] & ((1 << m) - 1))
            u, v = cv[h].copy(), cv[h + H].copy()
            cv[h], cv[h + H] = (u + v) % p, mm((u - v) % p, iw(m, j))
    # inverse low stages 8..5 by shfl lane^(1<<(m-4))
    for m in range(8, 4, -1):
        half, lb = 1 << m, 1 << (m - 4)
        w = iw(m, low & (half - 1))
        hi = (lane & lb) != 0
        x = cv[:, lane ^ lb]
        cv = np.where(hi[None, :], mm((x - cv) % p, w[None, :]), (cv + x) % p)
    inner = (bx << 1) | (lane & 1)
    xx = X
    oad = ((xx >> 5) * 1 + 0) * 32 + inner[None, :]   # K=1, pair 0
    assert (out[oad] == -1).all()
    out[oad] = cv
assert (out >= 0).all()

# --- crt_tc2 (r8f): CTA = 32-run, lane c = row
res = np.zeros(N, dtype=np.int64)
for blk in range(N // 32):
    start = blk * 32
    c = np.arange(32)
    r = out[((start >> 5) << 5) + (c & 31)]
    for m in range(4, -1, -1):
        half = 1 << m
        w = iw(m, c & (half - 1))
        x = r[c ^ half]
        r = np.where((c & half) != 0, mm((x - r) % p, w), (r + x) % p)
    res[start + c] = r

bad = int((res != RC).sum())
print("mismatches:", bad, "of", N)
assert bad == 0
print("R8F emulation OK")

# --- crt_tc2 (r8f v2): thread = limb, 32-point DIF stages 4..0 in registers
res2 = np.zeros(N, dtype=np.int64)
for blk in range(N // 32):
    v = [int(t) for t in out[blk * 32:(blk + 1) * 32]]
    tw = [int(iw(m, jj)) for m in range(5) for jj in range(1 << m)]  # (half-1)+jj
    for m in range(4, -1, -1):
        half = 1 << m
        for i in range(32):
            if i & half:
                continue
            u, x = v[i], v[i + half]
            v[i] = (u + x) % p
            v[i + half] = (u - x) * tw[(half - 1) + (i & (half - 1))] % p
    res2[blk * 32:(blk + 1) * 32] = v
assert (res2 == RC).all()
print("R8F v2 (thread=limb) OK")
