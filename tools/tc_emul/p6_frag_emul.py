"""P6: CPU model of the sparse radix-16 mma packing.

For one stage with 8 butterflies (i=0..7) and 8 groups (g=0..7), each product
v[g][i] * w[i] is computed with 4 mma calls (one per byte plane s of the
pre-scaled twiddle W_j[i] = w[i]*2^(8j) mod p).

mma rows = groups g (0..7, padded to 16), mma cols = butterfly i (0..7).
B[k=i*4+j][col] = byte_s(W_j[i]) if col==i else 0  (diagonal in k/col).
A[row=g][k=i*4+j] = byte_j(v[g][i]).

After 4 mma calls, S_s[g][i] = sum_j byte_j(v[g][i]) * byte_s(W_j[i]);
V[g][i] = sum_s S_s[g][i] << (8s) == v*w (mod p).
"""
import random

P = (1 << 31) - 19   # sample prime-ish modulus

def bytes4(x):
    return [(x >> (8*j)) & 0xff for j in range(4)]

def scalar_product(v, w, p):
    return (v * w) % p

def mma_sparse(v, w, p):
    # pre-scale W_j = w*2^(8j) mod p
    W = [(w * pow(2, 8*j, p)) % p for j in range(4)]
    Sb = [0]*4
    vb = bytes4(v)
    for s in range(4):
        Wb = [bytes4(W[j])[s] for j in range(4)]
        Sb[s] = sum(vb[j]*Wb[j] for j in range(4))
    V = sum(Sb[s] << (8*s) for s in range(4))
    return V % p

def main():
    random.seed(66)
    for t in range(5000):
        v = random.randrange(P)
        w = random.randrange(P)
        assert mma_sparse(v, w, P) == scalar_product(v, w, P), (t, v, w)
    print("P6 sparse byte-split: 5000 cases, 0 mismatches")

if __name__ == '__main__':
    main()
