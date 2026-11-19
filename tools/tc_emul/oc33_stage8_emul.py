"""OC33: stage-8 butterfly on two coalesced segments == TC29 shfl path."""
import random

def make(seed):
    random.seed(seed)
    while True:
        p = random.getrandbits(31) | (1 << 30) | 1
        if all(p % q for q in (3, 5, 7, 11, 13, 17, 19, 23)):
            return p

def mont(a, b, p, ni):
    t = a * b
    m = (t & 0xffffffff) * ni & 0xffffffff
    u = (t + m * p) >> 32
    return u - p if u >= p else u

def add(a, b, p):
    s = a + b
    return s - p if s >= p else s

def sub(a, b, p):
    return a - b if a >= b else a + p - b

def stage8(a, b, w, p, ni):
    """a = column c (lower), b = column c+256 (upper)."""
    v = mont(b, w, p, ni)
    return add(a, v, p), sub(a, v, p)

def main():
    for seed in range(20):
        p = make(seed)
        ni = pow(-pow(p, -1, 1 << 32), 1, 1 << 32)
        for _ in range(200):
            a, b, w = (random.randrange(p) for _ in range(3))
            # TC29 ef256: lane>=16 is upper, shfl(xa)=a, mont(av,b_w)=v
            # lane<16: av=lower=a, xa=upper=b -> add(a, w*b)
            # lane>=16: av=upper=b, xa=lower=a -> sub(a, w*b)
            lo_ref = add(a, mont(b, w, p, ni), p)
            up_ref = sub(a, mont(b, w, p, ni), p)
            # OC33 shared: sa=lower, sc=upper
            lo2, up2 = stage8(a, b, w, p, ni)
            assert (lo2, up2) == (lo_ref, up_ref), (seed, a, b, w, lo2, up2, lo_ref, up_ref)
    print("OC33 stage-8: 20 primes x 200 cases, OC33 == TC29 bit-exact")

if __name__ == '__main__':
    main()
