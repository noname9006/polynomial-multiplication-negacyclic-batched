"""OC31 tile-major rc layout: verify inv_low store <-> crt_tc2 load addressing.

inv_low (grid: tile=n/512, prime K, pair B; block 256):
  dst2[((pair*tiles + tile)*K + prime)*512 + pos] = value, pos in [0,512)

crt_tc2 (grid: start=32*blockIdx.x, prime j, pair; block 128):
  RC(j, coeff) = src[((pair*tiles + (start>>9))*K + j)*512 + (coeff & 511)]

For every CRT CTA row i = start + c (c = 0..31) and every prime j, RC(j,i)
must equal the value inv_low wrote for (pair, j, i).
"""
NT = 512
ROWS = 32

def inv_store(pair, tiles, tile, prime, pos, val, mem):
    mem[((pair * tiles + tile) * K + prime) * NT + pos] = val

def crt_load(pair, tiles, start, c, j, mem):
    i = start + c
    return mem[((pair * tiles + (start >> 9)) * K + j) * NT + (i & 511)]

K = 114
B = 3
n = 2048
tiles = n // NT
for pair in range(B):
    mem = {}
    for tile in range(tiles):
        for prime in range(K):
            for pos in range(NT):
                inv_store(pair, tiles, tile, prime, pos, pair * 1000000 + prime * 1000 + tile * NT + pos, mem)
    for start in range(0, n, ROWS):
        assert (start >> 9) < tiles
        for c in range(ROWS):
            i = start + c
            tile = i >> 9
            pos = i & 511
            for j in range(K):
                want = pair * 1000000 + j * 1000 + tile * NT + pos
                got = crt_load(pair, tiles, start, c, j, mem)
                assert got == want, (pair, start, c, j, got, want)
print("OC31 tile-major rc: B=%d tiles=%d K=%d - all CRT loads match inv_low stores" % (B, tiles, K))
