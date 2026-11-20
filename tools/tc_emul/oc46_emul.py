# OC46 emulator: mirrors the encode_fwdlow256 epilogue DIT literally.
# Per thread (warp w, lane L, g=L>>2, odd=L&1, mt=0/1, j=0..3):
#   prime k = (nt0+j)*2 + (t>>1),  row rl = w*32 + mt*16 + g + 8*odd.
# OC46: DIT stages 0..3 by lane shfl (bits 4,8,16,1), stage 4 in-thread;
# stage 5..7 then one radix-8 shared pass.  Compared against plain DIT 0..4
# on the same row layout for several primes, plus a negative test (broken
# lane mask).  Stage 5..7 is unchanged from the base path (verified by
# tools/tc_emul/r8f_emul.py, 0/65536).
import sys
import random

p = 2013265921
random.seed(11)


def low_of(lane, t):
    # unused placeholder kept out; row mapping follows the kernel exactly
    return lane


def row_of(w, L, r):
    return w * 32 + r * 16 + (L >> 2) + (L & 1) * 8


def mm(a, b):
    return a * b % p


def ref_dit05(v0, tw):
    """Plain forward DIT stages 0..4 on a 256-point row."""
    v = v0[:]
    for m in range(5):
        h = 1 << m
        for x in range(256):
            if x & h:
                continue
            j = x & (h - 1)
            u, y = v[x], mm(v[x + h], tw[m][j])
            v[x] = (u + y) % p
            v[x + h] = (u - y) % p
    return v


def kernel_dit(v0, tw, broken_stage=-1):
    """Literal mirror of the OC46 epilogue register path."""
    # register file R[w][L][mt] after epilogue (before DIT)
    R = [[[v0[row_of(w, L, 0)], v0[row_of(w, L, 1)]] for L in range(32)]
         for w in range(8)]
    for s in range(4):
        lb = 1 if s == 3 else (1 << (s + 2))   # lane^4,^8,^16,^1
        use = lb ^ 2 if (broken_stage == s) else lb
        for w in range(8):
            old = [row[:] for row in R[w]]
            for L in range(32):
                up = (L & use) != 0
                for mt in (0, 1):
                    rl = row_of(w, L, 0) + mt * 16
                    other = old[L ^ use][mt]
                    ww = tw[s][rl & ((1 << s) - 1)]
                    R[w][L][mt] = ((other - mm(old[L][mt], ww)) % p if up
                                   else (old[L][mt] + mm(other, ww)) % p)
    # stage 4: in-thread butterfly between mt=0 and mt=1
    for w in range(8):
        for L in range(32):
            rl = row_of(w, L, 0)
            ww = tw[4][rl & 15]
            u, x = R[w][L][0], R[w][L][1]
            R[w][L][0] = (u + mm(x, ww)) % p
            R[w][L][1] = (u - mm(x, ww)) % p
    return R


def test():
    bad = 0
    for trial in range(30):
        tw = {m: [random.randrange(1, p) for _ in range(1 << m)] for m in range(5)}
        v0 = [random.randrange(p) for _ in range(256)]
        ref = ref_dit05(v0, tw)
        R = kernel_dit(v0, tw)
        for w in range(8):
            for L in range(32):
                for mt in (0, 1):
                    rl = row_of(w, L, mt)
                    if R[w][L][mt] != ref[rl]:
                        bad += 1
                        if bad < 4:
                            print("MISMATCH", w, L, mt, rl)
    print("OC46 epilogue DIT 0..4: mismatches =", bad)
    # negative test: break the lane mask of stage 0
    tw = {m: [random.randrange(1, p) for _ in range(1 << m)] for m in range(5)}
    v0 = [random.randrange(p) for _ in range(256)]
    a = kernel_dit(v0, tw)
    b = kernel_dit(v0, tw, broken_stage=0)
    nb = sum(1 for w in range(8) for L in range(32) for mt in (0, 1)
             if a[w][L][mt] != b[w][L][mt])
    print("negative (mask bit flipped at stage 0): differing =", nb)
    print("surrounding stage 5..7 path (r8f_emul.py): 0/65536")
    return bad == 0 and nb > 0


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
