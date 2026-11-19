"""OC40: 2-thread carry over 56 u64 words + mod-q reduce vs naive chain."""
import random

def mont_reduce_sum(dg, l, q):
    # simplified: just return dg[0..l-1] mod q via repeated subtract (emulate fold)
    val = 0
    for i in range(l):
        val = (val << 32) | dg[i]
    return val % q

def carry_naive(xs, l):
    dg = [0] * (l + 2)
    carry = 0
    for w, x in enumerate(xs):
        carry += x
        dg[w] = carry & 0xffffffff
        carry >>= 32
    for w in range(len(xs), l + 2):
        dg[w] = carry & 0xffffffff
        carry >>= 32
    return dg

def carry_oc40(xs, l):
    words = len(xs)
    halfw = words // 2
    dg = [0] * (l + 2)
    segs = [[], []]
    big = [False, False]
    cout = [0, 0]
    for s in (0, 1):
        run = 0
        for j in range(halfw):
            x = xs[s * halfw + j]
            t = run + x
            dg[s * halfw + j] = t & 0xffffffff
            run = t >> 32
            if x >> 48:
                big[s] = True
        cout[s] = run | ((1 << 63) if big[s] else 0)
    replay = (cout[0] >> 63) != 0 or (cout[1] >> 63) != 0
    if replay:
        return carry_naive(xs, l)
    delta = cout[0]
    for j in range(halfw):
        if not delta:
            break
        sm = dg[halfw + j] + (delta & 0xffffffff)
        dg[halfw + j] = sm & 0xffffffff
        delta = (delta >> 32) + (sm >> 32)
    cin = cout[1] + delta
    for w in range(words, l + 2):
        dg[w] = cin & 0xffffffff
        cin >>= 32
    return dg

def main():
    random.seed(40)
    l = 55
    for trial in range(2000):
        mode = trial % 4
        xs = []
        for w in range(56):
            if mode == 0:
                x = random.getrandbits(64)
            elif mode == 1:
                x = random.choice([0, 1, (1 << 64) - 1, (1 << 48)])
            elif mode == 2:
                x = random.getrandbits(30) << 1
            else:
                x = random.getrandbits(48) - 1
            xs.append(x)
        a = carry_naive(xs, l)
        b = carry_oc40(xs, l)
        assert a == b, (trial, mode, [i for i in range(l + 2) if a[i] != b[i]][:5])
    print("OC40 carry: 2000 cases x 56 words, 0 mismatches")

if __name__ == '__main__':
    main()
