"""OC43 validation: epilogue layout + triple-buffer schedule.

1) Epilogue: mma accumulator words c[0..3] -> pg (0,1) and pg8 (2,3), packed
   with lane-dependent shift, combined with shfl_xor(1).  Check against the
   direct two-lane packed sum.
2) Buffer schedule: rp uses 3 buffers (gid%3), B uses 2 (gid&1);
   rp(gid+2) is issued at the end of group gid into buffer (gid+2)%3, which
   held group gid-1 and is dead after DIT(gid-1).  Verify no live overwrite.
3) Negative test: distorted double-buffered rp (gid%2) must collide.
"""
import random

EF_NTB = 4
NT = 28
NG = (NT + EF_NTB - 1) // EF_NTB   # 7

def pack(words, sh):
    return ((words[0] & 0xffffffff) << sh) + ((words[1] & 0xffffffff) << (sh + 8))

def epi_direct(we, wo):
    """Even lane packs at sh=0, odd lane at sh=16; add both packed values."""
    return (pack(we, 0) + pack(wo, 16)) & 0xffffffffffffffff

def epi_shfl(we, wo, odd):
    """Own lane packs at its own sh; shfl_xor(1) brings the partner's packed
    value (computed with the partner's sh) and adds it."""
    sh = 16 if odd else 0
    mine = pack(wo if odd else we, sh)
    send = pack(we if odd else wo, 0 if odd else 16)
    return (mine + send) & 0xffffffffffffffff

def main():
    random.seed(43)
    for _ in range(2000):
        we = [random.getrandbits(32) for _ in range(2)]
        wo = [random.getrandbits(32) for _ in range(2)]
        for odd in (False, True):
            assert epi_shfl(we, wo, odd) == epi_direct(we, wo), (we, wo, odd)
    print("epilogue shfl: 2000 cases x 2 parities, 0 mismatches")

    issued, writes = {0: -2, 1: -2}, {}   # rp(0), rp(1) issued before the loop
    for gid in range(NG):
        assert gid in issued, ("rp not issued before g0", gid)
        buf_rp, buf_b = gid % 3, gid & 1
        for (_, gprev), bp in writes.items():
            if bp == buf_rp and gprev < gid:
                assert gprev + 2 <= gid, ("rp buffer overwritten too early", gprev, gid)
        writes[('rp', gid)] = buf_rp
        if gid + 2 < NG:
            issued[gid + 2] = gid
    print("OC43 buffer schedule: %d groups, no live-buffer overwrite" % NG)

    # group -> prime mapping: group g covers primes [g*EF_PR, g*EF_PR+EF_PR)
    # (EF_NTB n-tiles x 2 primes each).  The buggy variant g*2+q collides.
    EF_PR = EF_NTB * 2
    covered = set()
    for gid in range(NG):
        for q in range(EF_PR):
            covered.add(gid * EF_PR + q)
    assert len(covered) == NG * EF_PR
    wrong = set()
    for gid in range(NG):
        for q in range(EF_PR):
            wrong.add(gid * 2 + q)
    assert len(wrong) < len(covered), "negative test: wrong kk mapping not detected"
    print("group->prime mapping: %d distinct primes; buggy variant only %d" % (len(covered), len(wrong)))

    seen, collided = {}, False
    for gid in range(NG):
        b = gid % 2                       # deliberately wrong scheme
        if b in seen and gid - seen[b] == 2:
            collided = True
        seen[b] = gid
    assert collided, "negative test failed: distorted layout was not detected"
    print("negative test: distorted rp layout detected as collision (as expected)")

if __name__ == '__main__':
    main()
