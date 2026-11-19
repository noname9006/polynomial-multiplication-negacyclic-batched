"""OC36: verify new sA layout gives the same 4 u32 A-fragment words as old."""
import random

K = 114
W = 120
NT = 28
KS = 15
SA = 132   # current stride
ROWS = 32
NTILE = 512

def old_store(sa, c, j, v):
    sa[c * SA + j] = v

def old_load4(sa, mt, g, ks, t):
    r0 = (mt * 16 + g) * SA + ks * 8 + t
    r1 = r0 + 8 * SA
    return (sa[r0], sa[r1], sa[r0 + 4], sa[r1 + 4])

# new layout: idx(j,c) = ((jq*32 + (c&7))*4 + t)*4 + (h*2 + (c>>3))
def idx(j, c):
    jq, jt = divmod(j, 8)
    h, t = divmod(jt, 4)
    return ((jq * 32 + (c & 7)) * 4 + t) * 4 + (h * 2 + (c >> 3))

def new_store(sa, c, j, v):
    sa[idx(j, c)] = v

def new_load4(sa, mt, g, ks, t):
    c = mt * 16 + g
    base = idx(ks * 8 + t, c) & ~3
    # the four words must be h=0/1, rhi=0/1 in this exact order
    return tuple(sa[base + i] for i in range(4))

def main():
    random.seed(36)
    sa_old = [random.randrange(1 << 32) for _ in range(ROWS * SA + 8)]
    sa_new = [0] * (16 * 32 * 4 * 4)
    for c in range(ROWS):
        for j in range(W):
            v = random.randrange(1 << 32)
            old_store(sa_old, c, j, v)
            new_store(sa_new, c, j, v)
    for mt in range(2):
        for g in range(8):          # lane>>2 => g in 0..7
            for ks in range(KS):
                for t in range(4):
                    o = old_load4(sa_old, mt, g, ks, t)
                    n = new_load4(sa_new, mt, g, ks, t)
                    assert o == n, (mt, g, ks, t, o, n)
    print("OC36 sA layout: all A fragments bit-identical (%d words)" % (2*8*KS*4))

if __name__ == '__main__':
    main()
