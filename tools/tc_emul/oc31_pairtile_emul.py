"""OC31: validate paired-row 256-tile mapping vs the reference 512-row low DIT.

Reference: one 512-tile, stages m=0..8, butterfly p <-> p ^ (1<<m),
twiddle TWI(m, p & ((1<<m)-1)) (TC22_MERGED: (1<<m)-1+j).

OC31: each CTA owns 256 tile rows r mapped to positions
  pos(r) = group + joff + (r & 127) + ((r >> 7) << 8), joff in {0,128}
so tile row r and r+128 are the stage-8 pair (j, j+256).  Stages 0..7 run on
tile rows (partner r ^ half, twiddle TWI(m, r & (half-1))); stage 8 is a
butterfly between rows j and j+128 with twiddle TWI(8, joff+j).
Output must be bit-identical to the reference.
"""
import random

def TWI(m, j):  # TC22_MERGED
    return (1 << m) - 1 + j

def bf(a, b, w, p):
    v = (b * w) % p
    return ((a + v) % p, (a - v) % p)

def low_dit_512(vals, tw, p):
    a = vals[:]
    for m in range(9):
        half = 1 << m
        for first in range(512):
            if first & half:
                continue
            w = tw[TWI(m, first & (half - 1))]
            a[first], a[first | half] = bf(a[first], a[first | half], w, p)
    return a

def low_dit_pairtile(vals, tw, p, joff, group):
    pos = lambda r: group + joff + (r & 127) + ((r >> 7) << 8)
    a = [vals[pos(r)] for r in range(256)]
    for m in range(8):
        half = 1 << m
        for first in range(256):
            if first & half:
                continue
            w = tw[TWI(m, first & (half - 1))]
            a[first], a[first | half] = bf(a[first], a[first | half], w, p)
    for j in range(128):
        w = tw[TWI(8, joff + j)]
        a[j], a[j + 128] = bf(a[j], a[j + 128], w, p)
    return pos, a

def main():
    random.seed(31)
    p = (1 << 31) - 1
    for trial in range(50):
        tw = [random.randrange(p) for _ in range(512)]
        vals = [random.randrange(p) for _ in range(512)]
        ref = low_dit_512(vals, tw, p)
        for tile in range(2):
            joff = tile * 128
            pos, got = low_dit_pairtile(vals, tw, p, joff, 0)
            for r in range(256):
                assert got[r] == ref[pos(r)], (trial, tile, r, got[r], ref[pos(r)])
    cover = []
    for tile in range(2):
        joff = tile * 128
        cover += [joff + (r & 127) + ((r >> 7) << 8) for r in range(256)]
    assert sorted(cover) == list(range(512)), sorted(cover)
    print("OC31 pair-tile low-DIT: 50 trials x 2 tiles x 256 positions, 0 mismatches")

if __name__ == "__main__":
    main()
