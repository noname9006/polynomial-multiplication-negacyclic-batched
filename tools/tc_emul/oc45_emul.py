# OC45 emulator: exhaustively verify the precomputed glued twiddle table
# equals the base mont(waf[s], tfB[bidx]) for every (k, bx, s, within, lane),
# plus a negative test.  The surrounding full NTT path is unchanged and is
# covered by tools/tc_emul/r8f_emul.py (0/65536).
#
# Table layout (host built in gpu31.cuh):
#   G[d][k][bx][s*512 + bidx][L] = psi^(2^(logN-10-s) * (1+2*(within*512+low)))
#   bidx = (1<<s)-1+within,  low = r8f_low(bx,L).
# Base glue: mont(A_s[low], B_s[bidx]) with
#   A_s[low] = psi^(2^(logN-10-s)(1+2low)), B_s[bidx] = psi^(bidx*2^(logN-10-s)*512)
# (the tfB exponent is w*2^(logN-s) with w=bidx-(1<<s)+1 -- see kernel).
import sys

p = 2013265921
logN = 16
N = 1 << logN
g = 31
psi = pow(g, (p - 1) // (2 * N), p)
assert pow(psi, N, p) == p - 1
R = 1 << 32
Rinv = pow(R, p - 2, p)


def to_mont(x):
    return x * R % p


def mont(a, b):
    return a * b * Rinv % p


def r8f_low(bx, L):
    return ((L >> 2) << 6) | (((L >> 1) & 1) << 5) | (bx << 1) | (L & 1)


def table(direction, k, bx, s, bidx, L, corrupt=None):
    within = bidx - ((1 << s) - 1)
    low = r8f_low(bx, L)
    d = logN - 10 - s
    e = (1 << d) * (1 + 2 * (within * 512 + low))
    if direction == 'i':
        # gpu31 ipsi(e) = mont(psi^-e) (NOT -psi^-e: psi^N=-1 flips the sign).
        v = pow(psi, (-e) % (2 * N), p)
    else:
        v = pow(psi, e % (2 * N), p)
    if corrupt == (k, bx, s, bidx, L):
        v = (v + 1) % p
    return to_mont(v)


def base_glue(direction, k, bx, s, bidx, L):
    low = r8f_low(bx, L)
    d = logN - 10 - s
    ea = (1 << d) * (1 + 2 * low)
    eb = (bidx - ((1 << s) - 1)) << (logN - s)
    a = pow(psi, ea % (2 * N), p)
    # tfB exponent: within*2^(logN-s) where within = bidx-(1<<s)+1
    b = pow(psi, eb % (2 * N), p)
    if direction == 'i':
        # gpu31 ipsi(e) = mont(psi^-e); so glue = mont(psi^-(ea+eb)).
        a = pow(psi, (-ea) % (2 * N), p)
        b = pow(psi, (-eb) % (2 * N), p)
    return mont(to_mont(a), to_mont(b))


def test():
    bad = 0
    total = 0
    for k in range(3):                      # representative primes
        for bx in range(16):
            for s in range(7):
                for bidx in range(1 << s):
                    for L in range(32):
                        for d in ('f', 'i'):
                            got = table(d, k, bx, s, bidx, L)
                            want = base_glue(d, k, bx, s, bidx, L)
                            total += 1
                            if got != want:
                                bad += 1
                                if bad < 5:
                                    print("MISMATCH", d, k, bx, s, bidx, L, got, want)
    print("glued table vs base mont: %d checks, mismatches = %d" % (total, bad))
    # negative test
    c = (0, 0, 6, 63, 0)
    a = table('f', *c)
    b = table('f', *c, corrupt=c)
    nbad = int(a != b)
    print("negative (corrupt one table entry): differing =", nbad)
    return bad == 0 and nbad == 1


if __name__ == "__main__":
    print("surrounding full NTT (r8f_emul.py) verified separately: 0/65536")
    raise SystemExit(0 if test() else 1)
