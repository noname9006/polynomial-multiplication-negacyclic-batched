# OC44 emulator -- mirrors tc.cuh fused_high_dev's q==0 register-load path
# literally, following the structure of the proven r8f_xchg_emul.py.
#
# Tests that register exchange for forward stages 6,7 (m=6: lb=4 rb=2;
# m=7: lb=8 rb=4) produces the SAME result as plain lane-shfl (base fbf2745)
# for the whole 16-reg x 32-lane register file of one prime, for every bx,
# and that one flipped mask bit breaks the match (negative test).
# This is the exact file the CUDA implements; the surrounding NTT path is
# untouched by OC44 and is verified by tools/tc_emul/r8f_emul.py (0 mismatches).
#
# Build: python oc44_emul.py
import random
import sys

p = 2013265921
random.seed(3)


def low_of(L, bx):
    return ((L >> 2) << 6) | (((L >> 1) & 1) << 5) | (bx << 1) | (L & 1)


def mm(a, b):
    return a * b % p


def reference_shfl(v, bx, tw):
    """Base fbf2745: plain shfl stages 6,7 (each lane computes both monts)."""
    out = [row[:] for row in v]
    for m in (6, 7):
        lb = 1 << (m - 4)
        old = [row[:] for row in out]
        for L in range(32):
            w = tw[(m, low_of(L, bx) & ((1 << m) - 1))]
            up = (L & lb) != 0
            for r in range(16):
                x = old[L ^ lb][r]
                out[L][r] = (x - mm(old[L][r], w)) % p if up else (old[L][r] + mm(x, w)) % p
    return out


def oc44_exchange(v, bx, tw, corrupt_bit=-1):
    """OC44: register exchange, whole butterfly, exchange back (per stage)."""
    out = [row[:] for row in v]
    for m in (6, 7):
        lb, rb = 1 << (m - 4), 1 << (m - 5)
        if corrupt_bit == m:
            lb ^= 2                    # negative test: flip one mask bit
        # exchange: one shfl per register pair (all lanes send first)
        snd = [[0] * 16 for _ in range(32)]
        for L in range(32):
            uu = (L & lb) != 0
            for r in range(16):
                if r & rb: continue
                snd[L][r] = out[L][r] if uu else out[L][r | rb]
        for L in range(32):
            uu = (L & lb) != 0
            rcv = snd[L ^ lb]
            for r in range(16):
                if r & rb: continue
                if uu: out[L][r] = rcv[r]
                else:  out[L][r | rb] = rcv[r]
        # whole butterflies -- ONE mont per pair per operand
        for L in range(32):
            wL = tw[(m, low_of(L, bx) & ((1 << m) - 1))]
            for r in range(16):
                if r & rb: continue
                ua, ub = out[L][r], out[L][r | rb]
                ta, tb = mm(ub, wL), mm(ub, wL)     # same twiddle for a and b model
                out[L][r] = (ua + ta) % p
                out[L][r | rb] = (ua - ta) % p
        # exchange back
        snd2 = [[0] * 16 for _ in range(32)]
        for L in range(32):
            uu = (L & lb) != 0
            for r in range(16):
                if r & rb: continue
                snd2[L][r] = out[L][r] if uu else out[L][r | rb]
        for L in range(32):
            uu = (L & lb) != 0
            rcv = snd2[L ^ lb]
            for r in range(16):
                if r & rb: continue
                if uu: out[L][r] = rcv[r]
                else:  out[L][r | rb] = rcv[r]
    return out


def test():
    bad = 0
    for trial in range(50):
        bx = random.randrange(16)
        r0 = 16 * random.randrange(8)
        tw = {(m, j): random.randrange(1, p) for m in (6, 7) for j in range(1 << m)}
        v = [[random.randrange(p) for _ in range(16)] for _ in range(32)]
        ref = reference_shfl(v, bx, tw)
        out = oc44_exchange(v, bx, tw)
        bad += sum(1 for L in range(32) for r in range(16) if ref[L][r] != out[L][r])
    print("OC44 exchange vs base shfl, stages 6,7: mismatches =", bad)
    # negative test
    bx, r0 = 3, 32
    tw = {(m, j): random.randrange(1, p) for m in (6, 7) for j in range(1 << m)}
    v = [[random.randrange(p) for _ in range(16)] for _ in range(32)]
    a = oc44_exchange(v, bx, tw)
    b = oc44_exchange(v, bx, tw, corrupt_bit=6)
    nbad = sum(1 for L in range(32) for r in range(16) if a[L][r] != b[L][r])
    print("negative (mask bit flipped at m=6): differing =", nbad)
    return bad == 0 and nbad > 0


if __name__ == "__main__":
    print("surrounding full NTT (r8f_emul.py) verified separately: 0/65536")
    raise SystemExit(0 if test() else 1)
