p = 'tc.cuh'
s = open(p, encoding='utf-8').read()

# 1. flag
anchor = '#ifndef TC30_CPA\n#define TC30_CPA 1\n#endif\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''// OC43: triple-buffer sR/sP/sW in encode_fwdlow256.  The next-next group's
// twiddles/prime/weight are issued at the END of the current group, so the
// cp.async wait at g0 overlaps a full group of mma/epi/DIT work.
// Flag 0 = exact copy of master.  Static shared 48,192 B <= 49,152.
#ifndef OC43_ENCGEMM
#define OC43_ENCGEMM 1
#endif
''')

# 2. shared declarations
old = '''#if TC30_CPA
    __shared__ __align__(16) u32 sRb[2][EF_PR][E2_ROWS];
    __shared__ __align__(16) uint2 sBb[2][EF_NTB * ENC_KS_MAX * 32];
    __shared__ __align__(16) gpu31::Prime sPb[2][EF_PR];    // TC32
    __shared__ __align__(16) gpu31::Weight sWb[2][EF_PR];
#else'''
new = '''#if OC43_ENCGEMM
    __shared__ __align__(16) u32 sR3[3][EF_PR][E2_ROWS];
    __shared__ __align__(16) uint2 sB2[2][EF_NTB * ENC_KS_MAX * 32];
    __shared__ __align__(16) gpu31::Prime sP3[3][EF_PR];
    __shared__ __align__(16) gpu31::Weight sW3[3][EF_PR];
#elif TC30_CPA
    __shared__ __align__(16) u32 sRb[2][EF_PR][E2_ROWS];
    __shared__ __align__(16) uint2 sBb[2][EF_NTB * ENC_KS_MAX * 32];
    __shared__ __align__(16) gpu31::Prime sPb[2][EF_PR];    // TC32
    __shared__ __align__(16) gpu31::Weight sWb[2][EF_PR];
#else'''
assert s.count(old) == 1
s = s.replace(old, new)

# 3. issue lambda: split into issue_rp / issue_b under OC43
start = s.index('#if TC30_CPA\n    // TC30: group twiddles/B fragments arrive by cp.async')
end = s.index('    for (unsigned nt0 = 0; nt0 < NT; nt0 += EF_NTB) {', start)
oldblock = s[start:end]
assert 'issue(0, 0);' in oldblock
newblock = '''#if OC43_ENCGEMM
    // OC43: twiddles/prime/weight (rp) go into 3 buffers; B fragments stay
    // double-buffered.  rp for group g+2 is issued at the end of group g.
    auto issue_rp = [&](unsigned ng, unsigned buf) {
        for (unsigned idx = tid; idx < EF_PR * E2_ROWS; idx += 256) {
            const unsigned q = idx / E2_ROWS, r = idx % E2_ROWS;
            const unsigned kk = ng * 2 + q;
            const bool ok = kk < K;
            const u32* gp = packed + (ok ? u64(kk) * TC_LT + r : 0);
            const unsigned sa = unsigned(__cvta_generic_to_shared(&sR3[buf][q][r]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 4 : 0) : "memory");
        }
        if (tid < 2 * EF_PR) {
            const unsigned q = tid & (EF_PR - 1);
            const unsigned kk = ng * 2 + q;
            const bool ok = kk < K;
            if (tid < EF_PR) {
                const gpu31::Prime* gp = primes + (ok ? kk : 0);
                const unsigned sa = unsigned(__cvta_generic_to_shared(&sP3[buf][q]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 16 : 0) : "memory");
            } else {
                const gpu31::Weight* gp = weights + (ok ? std::size_t(kk) * l : 0);
                const unsigned sa = unsigned(__cvta_generic_to_shared(&sW3[buf][q]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 8 : 0) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    auto issue_b = [&](unsigned ng) {
        uint2* dstb = sB2[ng & 1u];
        for (unsigned idx = tid; idx < EF_NTB * KS * 32; idx += 256) {
            const unsigned j = idx / (KS * 32), rem = idx - j * (KS * 32);
            const unsigned nt = ng * EF_NTB + j;
            const bool ok = nt < NT;
            const uint2* gp = frag + (ok ? std::size_t(nt) * KS * 32 + rem : 0);
            const unsigned sa = unsigned(__cvta_generic_to_shared(&dstb[idx]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 8 : 0) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    issue_rp(0, 0);
    issue_rp(1, 1);
    issue_b(0);
#elif ''' + oldblock[len('#if TC30_CPA\n'):]
s = s[:start] + newblock + s[end:]

# 4. g0 buffer selection
old = '''#if TC30_CPA
        const unsigned cb = (nt0 / EF_NTB) & 1;
        u32 (*sR)[E2_ROWS] = sRb[cb];
        uint2* sB = sBb[cb];
#if TC32_PW
        const gpu31::Prime* sP = sPb[cb];
        const gpu31::Weight* sW = sWb[cb];
#endif
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (nt0 + EF_NTB < NT) issue(nt0 + EF_NTB, cb ^ 1);
#else'''
new = '''#if OC43_ENCGEMM
        const unsigned gid = nt0 / EF_NTB;
        u32 (*sR)[E2_ROWS] = sR3[gid % 3];
        uint2* sB = sB2[gid & 1u];
        const gpu31::Prime* sP = sP3[gid % 3];
        const gpu31::Weight* sW = sW3[gid % 3];
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (nt0 + EF_NTB < NT) issue_b(gid + 1);
#elif TC30_CPA
        const unsigned cb = (nt0 / EF_NTB) & 1;
        u32 (*sR)[E2_ROWS] = sRb[cb];
        uint2* sB = sBb[cb];
#if TC32_PW
        const gpu31::Prime* sP = sPb[cb];
        const gpu31::Weight* sW = sWb[cb];
#endif
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (nt0 + EF_NTB < NT) issue(nt0 + EF_NTB, cb ^ 1);
#else'''
assert s.count(old) == 1
s = s.replace(old, new)

# 5. issue rp(gid+2) at end of group
old = '''        TCP_T(g5);
        pe[0] += g1 - g0;'''
new = '''        TCP_T(g5);
#if OC43_ENCGEMM
        if (gid + 2 < (NT + EF_NTB - 1) / EF_NTB) issue_rp(gid + 2, (gid + 2) % 3);
#endif
        pe[0] += g1 - g0;'''
assert s.count(old) == 2
i = s.rfind(old)          # the second one is inside encode_fwdlow256
s = s[:i] + new + s[i + len(old):]

open(p, 'w', encoding='utf-8').write(s)
print('patched OC43')
