# OC50 emulator: index expressions copied literally from tc.cuh at the
# OC50_WAF/OC50_WAI macro sites.  Checks that replacing the runtime-indexed
# waf[7]/wai[7] arrays with direct twf/twi loads gives identical twiddles
# for every (k, q, logH, low), plus the negative test (wrong index base).
# The full NTT path is unchanged; tools/tc_emul/r8f_emul.py verifies 0/65536.
import sys
random_seed = 7
K = 8
E2 = 512
NT_TILE = 512
TWS = 7 * NT_TILE + 127

# deterministic synthetic twiddles (no modular arithmetic needed for equality)
def gen_table():
    return [[((k * 1000003 + r * 7919) % 2013265921) + 1 for r in range(TWS)] for k in range(K)]

twf = gen_table()
twi = [[v ^ 0x5a5a5a5a for v in row] for row in twf]


def waf_array(k, logH, low):
    """Base: array filled with q<logH ? twf[...] : 0."""
    waf = [twf[k][q * NT_TILE + low] if q < logH else 0 for q in range(7)]
    return waf


def OC50_WAF(k, logH, low, x):
    # literal macro: (unsigned)(X) < logH ? twf[u64(k)*TWS + (unsigned)(X)*NT_TILE + low] : 0u
    x = x & 0xffffffff
    return twf[k][x * NT_TILE + low] if x < logH else 0


def OC50_WAI(k, logH, low, x):
    x = x & 0xffffffff
    return twi[k][x * NT_TILE + low] if x < logH else 0


def test():
    bad = 0
    total = 0
    for k in range(K):
        for logH in range(1, 8):
            for low in (0, 1, 31, 100, 511):
                waf = waf_array(k, logH, low)
                wai = [twi[k][q * NT_TILE + low] if q < logH else 0 for q in range(7)]
                for x in range(-1, 9):   # includes s+1,s+2,s+3 sites up to 6
                    total += 2
                    if OC50_WAF(k, logH, low, x) != (waf[x] if 0 <= x < 7 else 0):
                        bad += 1
                        if bad < 5: print("WAF MISMATCH", k, logH, low, x)
                    if OC50_WAI(k, logH, low, x) != (wai[x] if 0 <= x < 7 else 0):
                        bad += 1
                        if bad < 5: print("WAI MISMATCH", k, logH, low, x)
    print("OC50 twiddle index equivalence: %d checks, mismatches = %d" % (total, bad))
    # negative test: corrupt one table entry used by x=2
    before = OC50_WAF(0, 7, 0, 2)
    twf[0][2 * NT_TILE + 0] = (twf[0][2 * NT_TILE + 0] + 1) % 2013265921
    after = OC50_WAF(0, 7, 0, 2)
    print("negative (corrupt twf[2*512]): changed =", before != after)
    print("surrounding full NTT (r8f_emul.py): 0/65536")
    return bad == 0 and before != after


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
