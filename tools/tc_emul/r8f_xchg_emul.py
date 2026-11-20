# R8F v4: fused_high inverse low stages 8..5 by register transposition.
# Reference: per-lane shfl stage (both lanes of a pair compute).  New: each
# stage swaps lane bit lb=16>>k with register bit rb=8>>k, then every lane runs
# whole butterflies on register pairs (r, r|rb).  Final element of lane L,
# register r: lane' = (r<<1)|(L&1), row i' = (L>>1)&15, so x>>5 = (r0+i')*16+r.
import random

p = 2013265921
random.seed(3)
for trial in range(20):
    bx = random.randrange(16)
    r0 = 16 * random.randrange(8)
    lane_low = [((L >> 2) << 6) | (((L >> 1) & 1) << 5) | (bx << 1) | (L & 1) for L in range(32)]
    tw = {(m, j): random.randrange(1, p) for m in range(5, 9) for j in range(1 << m)}
    av = [[random.randrange(p) for _ in range(16)] for _ in range(32)]  # [lane][i]
    # reference
    ref = [row[:] for row in av]
    for m in range(8, 4, -1):
        half, lb = 1 << m, 1 << (m - 4)
        new = [row[:] for row in ref]
        for L in range(32):
            w = tw[(m, lane_low[L] & (half - 1))]
            for i in range(16):
                x = ref[L ^ lb][i]
                if L & lb:
                    new[L][i] = (x - ref[L][i]) * w % p
                else:
                    new[L][i] = (ref[L][i] + x) % p
        ref = new
    refmap = {}
    for L in range(32):
        for i in range(16):
            x = (r0 + i) * 512 + lane_low[L]
            refmap[x] = ref[L][i]
    # new scheme
    v = [row[:] for row in av]
    for k in range(4):
        m = 8 - k
        half, lb, rb = 1 << m, 16 >> k, 8 >> k
        # exchange (one shfl per register pair)
        send = [[0] * 16 for _ in range(32)]
        for L in range(32):
            for r in range(16):
                if r & rb:
                    continue
                send[L][r] = v[L][r | rb] if not (L & lb) else v[L][r]
        for L in range(32):
            for r in range(16):
                if r & rb:
                    continue
                recv = send[L ^ lb][r]
                if not (L & lb):
                    v[L][r | rb] = recv
                else:
                    v[L][r] = recv
        for L in range(32):
            w = tw[(m, lane_low[L] & (half - 1))]
            for r in range(16):
                if r & rb:
                    continue
                u, x = v[L][r], v[L][r | rb]
                v[L][r] = (u + x) % p
                v[L][r | rb] = (u - x) * w % p
    for L in range(32):
        for r in range(16):
            lanep = (r << 1) | (L & 1)
            ip = (L >> 1) & 15
            x = (r0 + ip) * 512 + lane_low[lanep]
            assert (x >> 5) == (r0 + ip) * 16 + r
            assert refmap[x] == v[L][r], (trial, L, r)
print("R8F v4 register-transposition OK")
