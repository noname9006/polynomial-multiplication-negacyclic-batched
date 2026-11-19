"""OC32: mont-by-CRT-constant commutes with the inverse NTT.

Model the exact gpu31 arithmetic: residues/twiddles live in Montgomery form
(x = X*R mod p), mont(a,b) = a*b*R^-1 mod p, butterfly uses add/sub/mont.

Path A (current): inverse NTT on residues, then mont(a, slotc).
Path B (OC32):    mont(residue, slotc) first, then the same inverse NTT.
The two must be bit-identical for every prime.
"""
import random

R = 1 << 32

def make_prime(seed):
    random.seed(seed)
    while True:
        p = random.getrandbits(31) | (1 << 30) | 1
        if all(p % q for q in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)):
            return p

def mont(a, b, p, neg_inv):
    t = a * b
    m = (t & 0xffffffff) * neg_inv & 0xffffffff
    u = (t + m * p) >> 32
    return u - p if u >= p else u

def add(a, b, p):
    s = a + b
    return s - p if s >= p else s

def sub(a, b, p):
    return a - b if a >= b else a + p - b

def inv_ntt(a, tw, p, neg_inv):
    a = a[:]
    for m in range(8, -1, -1):
        half = 1 << m
        for first in range(512):
            if first & half:
                continue
            j = first & (half - 1)
            w = tw[(half - 1) + j]
            u, v = a[first], a[first | half]
            a[first] = add(u, v, p)
            a[first | half] = mont(sub(u, v, p), w, p, neg_inv)
    return a

def main():
    for seed in range(8):
        p = make_prime(seed + 1)
        neg_inv = pow(-pow(p, -1, R), 1, R)
        slotc = random.randrange(p)          # CRT constant in Montgomery form
        tw = [random.randrange(p) for _ in range(512)]
        vals = [random.randrange(p) for _ in range(512)]
        # Path A: inverse NTT first, then mont by slotc
        a = inv_ntt(vals, tw, p, neg_inv)
        path_a = [mont(x, slotc, p, neg_inv) for x in a]
        # Path B: preweight each residue, then the same inverse NTT
        pre = [mont(x, slotc, p, neg_inv) for x in vals]
        path_b = inv_ntt(pre, tw, p, neg_inv)
        assert path_a == path_b, (seed, [i for i in range(512) if path_a[i] != path_b[i]][:5])
    print("OC32 preweight: 8 primes x 512 points, path A == path B bit-exact")

if __name__ == "__main__":
    main()
