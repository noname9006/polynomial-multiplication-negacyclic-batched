import re
p = 'tc.cuh'
s = open(p, encoding='utf-8').read()

# 1. flag
anchor = "#ifndef TC29_EF256\n#define TC29_EF256 1\n#endif\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + """// OC33: fused_high owns two contiguous 32-column segments per CTA
// (grid.x 16 -> 8), both coalesced 128 B/warp.  Low stage 8 becomes an
// in-shared butterfly between the segments; the two segments are then
// processed sequentially through the same NTT pipeline.  0 = TC29 path.
#ifndef OC33_COAL
#define OC33_COAL 0
#endif
""")

# 2. disable OC12_IO fast path in the coalesced variant
s = s.replace('#if OC10_R16 && OC12_IO', '#if OC10_R16 && OC12_IO && !OC33_COAL')
assert s.count('#if OC10_R16 && OC12_IO && !OC33_COAL') == 5

# 3. shared pointers
old = """    u32* sa = sm;
    u32* sb = sm + HC * S;
    const unsigned lane = threadIdx.x, rw = threadIdx.y;
    const unsigned low = ef256 ? (blockIdx.x * 16 + (lane & 15) + ((lane & 16) << 4))
                               : blockIdx.x * NT_COLS + lane;
    // TC29: twiddle of low stage 8 for pair (j, j+256), j = low & 255
    const u32 w8 = ef256 ? pkf[u64(blockIdx.y) * TC_LT + TWI(8, low & 255)] : 0u;
"""
new = """    u32* sa = sm;
    u32* sb = sm + HC * S;
#if OC33_COAL
    u32* sc = sm + 2 * HC * S;   // segment 1 (columns low+256)
    u32* sd = sm + 3 * HC * S;
#endif
    const unsigned lane = threadIdx.x, rw = threadIdx.y;
#if OC33_COAL
    const unsigned low = blockIdx.x * NT_COLS + lane;
    const u32 w8 = pkf[u64(blockIdx.y) * TC_LT + TWI(8, low)];
#else
    const unsigned low = ef256 ? (blockIdx.x * 16 + (lane & 15) + ((lane & 16) << 4))
                               : blockIdx.x * NT_COLS + lane;
    // TC29: twiddle of low stage 8 for pair (j, j+256), j = low & 255
    const u32 w8 = ef256 ? pkf[u64(blockIdx.y) * TC_LT + TWI(8, low & 255)] : 0u;
#endif
"""
assert s.count(old) == 1
s = s.replace(old, new)

# 4. load loop: add second segment + stage-8 butterfly
old = """        for (unsigned h = rw; h < HC; h += 8) {
            sa[h * S + lane] = ra[plane + u64(h) * NT_TILE + low];
            sb[h * S + lane] = rb[plane + u64(h) * NT_TILE + low];
        }
        __syncthreads();
"""
new = """        for (unsigned h = rw; h < HC; h += 8) {
            sa[h * S + lane] = ra[plane + u64(h) * NT_TILE + low];
            sb[h * S + lane] = rb[plane + u64(h) * NT_TILE + low];
#if OC33_COAL
            sc[h * S + lane] = ra[plane + u64(h) * NT_TILE + low + 256];
            sd[h * S + lane] = rb[plane + u64(h) * NT_TILE + low + 256];
#endif
        }
        __syncthreads();
#if OC33_COAL
        // Low stage 8: rows h of segment 0 (column low) and segment 1
        // (column low+256).  Twiddle depends only on the low column index.
        for (unsigned h = rw; h < HC; h += 8) {
            u32 ua = sa[h * S + lane], ub = sc[h * S + lane];
            u32 va = sb[h * S + lane], vb = sd[h * S + lane];
            sc[h * S + lane] = gpu31::mont(gpu31::sub(ua, ub, p.p), w8, p);
            sa[h * S + lane] = gpu31::add(ua, ub, p.p);
            sd[h * S + lane] = gpu31::mont(gpu31::sub(va, vb, p.p), w8, p);
            sb[h * S + lane] = gpu31::add(va, vb, p.p);
        }
        __syncthreads();
#endif
"""
assert s.count(old) == 1
s = s.replace(old, new)

# 5. wrap the main pipeline in a two-segment loop
start = s.index('    TCP_T(f1);\n', s.index('fused_high('))
end_marker = '    TCP_ADD(8, f3, f4); TCP_ADD(9, f4, f5);\n}'
end = s.index(end_marker, start)
body = s[start:end]
body33 = re.sub(r'\bsa\b', 'sA', body)
body33 = re.sub(r'\bsb\b', 'sB', body33)
body33 = re.sub(r'\blow\b', 'lo', body33)
# compute per-segment low twiddles at the top of the loop
prefix = """    for (unsigned seg = 0; seg < 2; ++seg) {
        u32* sA = seg ? sc : sa;
        u32* sB = seg ? sd : sb;
        const unsigned lo = low + seg * 256;
#pragma unroll
        for (unsigned q = 0; q < 7; ++q) {
            waf[q] = q < logH ? twf[u64(k) * TWS + q * NT_TILE + lo] : 0u;
            wai[q] = q < logH ? twi[u64(k) * TWS + q * NT_TILE + lo] : 0u;
        }
"""
newblock = "#if OC33_COAL\n" + prefix + body33 + "    }\n#else\n" + body + "#endif\n"
s = s[:start] + newblock + s[end:]

open(p, 'w', encoding='utf-8').write(s)
print('patched tc.cuh, OC33_COAL wrap length', len(body33))
