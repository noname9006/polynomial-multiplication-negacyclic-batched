"""H2-sparse: full radix-16 DIT bit-exact emulator with IMMA byte-split products.

Key correctness point (found by this emulator): P6 measured only mma THROUGHPUT
using a Shoup-style pre-scaled packing (W_j = w*2^(8j) mod p).  mont(b,w) needs
the EXACT 64-bit product a*b, so the IMMA must compute the byte CONVOLUTION:
  a = sum_i a_i 2^(8i), b = sum_j b_j 2^(8j)
  c_s = sum_{i+j=s} a_i b_j   (s = 0..6), each < 2^18
  a*b = sum_s c_s 2^(8s)  (with carries between planes)
Each mma_u8 call computes one plane c_s across 8 groups x 16 butterflies.
Then Montgomery reduce: m = low32(a*b)*neg_inv, u = (a*b + m*p)>>32.

Verify the full radix-16 DIT bit-exact against scalar mont, 8 primes x 200 groups.
"""
import random

R = 1 << 32

def make_prime(seed):
    random.seed(seed)
    while True:
        p = random.getrandbits(31) | (1 << 30) | 1
        if all(p % q for q in (3,5,7,11,13,17,19,23,29,31,37,41)):
            return p

def mont_scalar(a, b, p, neg_inv):
    t = a * b
    m = (t & 0xffffffff) * neg_inv & 0xffffffff
    u = (t + m * p) >> 32
    return u - p if u >= p else u

def bytes4(x):
    return [(x >> (8*j)) & 0xff for j in range(4)]

def mont_imma(a, b, p, neg_inv):
    """Exact a*b via byte-convolution (7 mma planes), then Montgomery reduce."""
    ab = bytes4(a)
    bb = bytes4(b)
    c = [0]*7
    for i in range(4):
        for j in range(4):
            c[i+j] += ab[i] * bb[j]
    # carry-propagate the 8-bit planes into a 64-bit integer
    prod = 0
    carry = 0
    for s in range(6):
        v = c[s] + carry
        prod |= (v & 0xff) << (8*s)
        carry = v >> 8
    prod |= (c[6] + carry) << 48
    prod &= 0xffffffffffffffff
    assert prod == a * b, (a, b, prod, a*b)
    m = (prod & 0xffffffff) * neg_inv & 0xffffffff
    u = (prod + m * p) >> 32
    return u - p if u >= p else u

def bf(a, b, w, p, neg_inv, mont):
    v = mont(b, w, p, neg_inv)
    return ((a + v) % p, (a - v) % p)

def radix16_dit(v, tw, p, neg_inv, mont):
    v = v[:]
    t1 = tw[0]
    t2 = [tw[1], tw[2]]
    t4 = [tw[3], tw[4], tw[5], tw[6]]
    t8 = [tw[7+i] for i in range(8)]
    for i in range(0, 16, 2):
        v[i], v[i+1] = bf(v[i], v[i+1], t1, p, neg_inv, mont)
    for k in range(2):
        for i in range(k, 16, 4):
            v[i], v[i+2] = bf(v[i], v[i+2], t2[k], p, neg_inv, mont)
    for k in range(4):
        for i in range(k, 16, 8):
            v[i], v[i+4] = bf(v[i], v[i+4], t4[k], p, neg_inv, mont)
    for i in range(8):
        v[i], v[i+8] = bf(v[i], v[i+8], t8[i], p, neg_inv, mont)
    return v

def main():
    for seed in range(8):
        p = make_prime(seed + 1)
        neg_inv = pow(-pow(p, -1, R), 1, R)
        for _ in range(200):
            tw = [random.randrange(p) for _ in range(15)]
            v = [random.randrange(p) for _ in range(16)]
            ref = radix16_dit(v, tw, p, neg_inv, mont_scalar)
            got = radix16_dit(v, tw, p, neg_inv, mont_imma)
            assert ref == got, (seed, v, tw, [i for i in range(16) if ref[i]!=got[i]])
    print("H2 full radix-16: 8 primes x 200 groups, IMMA byte-convolution mont == scalar, 0 mismatches")

if __name__ == '__main__':
    main()
