// oc51_emul.cpp — validate OC51 T2 kernel: 2 tiles/CTA, one B fragment feeds
// 4 mma, epilogue/carry/store per tile reusing ONE sX.  Mirrors the kernel's
// index arithmetic line-for-line; compares against master P32 and naive big-int.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "gpu31.cuh"
#include "tc.cuh"
using u32=std::uint32_t; using u64=std::uint64_t;
namespace gpu31 { u64 shared[4]; u32 stage[4]; u32 data[4]; }
namespace tc { unsigned char smem_raw[16]; u32 sm[4]; }

static void mma_emul(const u32 A[32][4], const u32 Bf[32][2], long long C[16][8]) {
    unsigned char a[16][32] = {}, b[32][8] = {};
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; ++i) {
            a[g][t*4+i]      = A[lane][0] >> (8*i);
            a[g+8][t*4+i]    = A[lane][1] >> (8*i);
            a[g][16+t*4+i]   = A[lane][2] >> (8*i);
            a[g+8][16+t*4+i] = A[lane][3] >> (8*i);
            b[t*4+i][g]      = Bf[lane][0] >> (8*i);
            b[16+t*4+i][g]   = Bf[lane][1] >> (8*i);
        }
    }
    for (int r = 0; r < 16; ++r)
        for (int cc = 0; cc < 8; ++cc)
            for (int k = 0; k < 32; ++k) C[r][cc] += (long long)a[r][k] * b[k][cc];
}
static bool g_epi_dbg=false;
static void epi_emul(const long long C[16][8], unsigned nt, unsigned SX,
                     unsigned mt, u64* sX) {
    for (int lane = 0; lane < 32; ++lane) {
        int t = lane & 3, g = lane >> 2;
        unsigned sh = (t & 1) ? 16u : 0u;
        bool odd = t & 1;
        int pt = lane ^ 1; int tp = pt & 3;
        unsigned shp = (tp & 1) ? 16u : 0u;
        u64 pg  = (u64(u32(C[g][2*t])) << sh)   + (u64(u32(C[g][2*t+1])) << (sh+8));
        u64 pg8 = (u64(u32(C[g+8][2*t])) << sh) + (u64(u32(C[g+8][2*t+1])) << (sh+8));
        u64 spg = (u64(u32(C[g][2*tp])) << shp) + (u64(u32(C[g][2*tp+1])) << (shp+8));
        u64 spg8= (u64(u32(C[g+8][2*tp])) << shp)+ (u64(u32(C[g+8][2*tp+1])) << (shp+8));
        u64 tot = (odd ? pg8 : pg) + (odd ? spg8 : spg);
        unsigned row = mt*16 + g + (odd ? 8u : 0u);
        sX[row*SX + nt*2 + (t >> 1)] = tot;
    }
}
static void carry_seq(const u64* xs, u32* dg, unsigned words, unsigned l2) {
    u64 carry = 0;
    for (unsigned w = 0; w < words; ++w) {
        carry += xs[w];
        dg[w] = u32(carry);
        carry >>= 32;
    }
    for (unsigned w = words; w < l2; ++w) { dg[w] = u32(carry); carry >>= 32; }
}
static void ref_row(const u32* terms, const u32* W, unsigned pitch, unsigned L,
                    unsigned bits, u32 gap, const u32* q, u32* out) {
    std::vector<u32> acc(L+2, 0);
    for (unsigned j = 0; j < pitch; ++j) {
        u64 cc = 0;
        for (unsigned i = 0; i < L+2; ++i) {
            u64 v = cc + acc[i] + (i < L ? u64(terms[j]) * W[i*pitch+j] : 0);
            acc[i] = u32(v); cc = v >> 32;
        }
    }
    gpu31::reduce_sum<1>(acc.data(), L, bits, gap, q);
    for (unsigned i = 0; i < L; ++i) out[i] = acc[i];
}
struct Shape{unsigned K,L,pitch,KS,NT,SA,SX;};
static Shape mk(unsigned K,unsigned L){Shape s{};s.K=K;s.L=L;s.pitch=K+1;
    unsigned ks,nt;std::vector<u32> w(std::size_t(L)*s.pitch,0);
    tc::bake_crt(w,L,s.pitch,ks,nt);s.KS=ks;s.NT=nt;s.SA=tc::crt_sa_stride(ks);s.SX=2*nt+1;return s;}

static std::vector<u64> g_p32;
static int run(const Shape& sh,unsigned rows,unsigned mode,std::mt19937_64& rng,
               unsigned bits,u32 gap,const u32* q,const char* name){
    const unsigned K=sh.K,L=sh.L,pitch=sh.pitch,SA=sh.SA,SX=sh.SX,KS=sh.KS,NT=sh.NT;
    std::vector<u32> W(std::size_t(L)*pitch);
    for(auto& x:W)x=u32(rng());
    unsigned ks=KS,nt=NT; auto frag=tc::bake_crt(W,L,pitch,ks,nt);
    std::vector<u32> T(std::size_t(rows)*SA,0);
    for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<pitch;++j)T[r*SA+j]=u32(rng())&0x7fffffffu;
    if(mode==1)for(auto& x:T)x=0;
    if(mode==2)for(auto& x:T)x=0xffffffffu;
    if(mode==3)for(unsigned r=0;r<rows;++r){T[r*SA]=0x7fffffffu;T[r*SA+1]=0xffffffffu;}
    if(mode==4)for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<pitch;++j)T[r*SA+j]=(j&1)?0xffffffffu:0x7fffffffu;
    if(mode==5)for(auto& x:T)x=0xffffffffu;

    // master P32 (baseline)
    std::vector<u32> out32(std::size_t(rows)*L,0);
    for(unsigned cta=0;cta*32<rows;++cta){const unsigned r0=cta*32,nr=(rows-r0<32)?rows-r0:32;
        std::vector<u64> sx(std::size_t(32)*SX,0);
        for(unsigned warp=0;warp<4;++warp)for(unsigned nt0=warp;nt0<NT;nt0+=8){
            unsigned ntA=nt0,ntB=nt0+4;bool hasB=ntB<NT;
            long long C[2][2][16][8]={};
            for(unsigned k=0;k<KS;++k){u32 B0[32][2],B1[32][2];
                for(int lane=0;lane<32;++lane){
                    B0[lane][0]=frag[((std::size_t(ntA)*KS+k)*32+lane)*2];
                    B0[lane][1]=frag[((std::size_t(ntA)*KS+k)*32+lane)*2+1];
                    if(hasB){B1[lane][0]=frag[((std::size_t(ntB)*KS+k)*32+lane)*2];
                        B1[lane][1]=frag[((std::size_t(ntB)*KS+k)*32+lane)*2+1];
                    }else B1[lane][0]=B1[lane][1]=0;}
                for(unsigned mt=0;mt<2;++mt){u32 A[32][4];
                    for(int lane=0;lane<32;++lane){int g=lane>>2,t=lane&3;
                        unsigned rr=r0+mt*16+g,rr8=rr+8;bool ok=rr<rows,ok8=rr8<rows;
                        A[lane][0]=ok?T[rr*SA+k*8+t]:0;A[lane][1]=ok8?T[rr8*SA+k*8+t]:0;
                        A[lane][2]=ok?T[rr*SA+k*8+4+t]:0;A[lane][3]=ok8?T[rr8*SA+k*8+4+t]:0;}
                    mma_emul(A,B0,C[mt][0]);if(hasB)mma_emul(A,B1,C[mt][1]);}}
            for(unsigned mt=0;mt<2;++mt){epi_emul(C[mt][0],ntA,SX,mt,sx.data());
                if(hasB)epi_emul(C[mt][1],ntB,SX,mt,sx.data());}}
        if(cta==0) g_p32.assign(sx.begin(),sx.begin()+32*SX);
        if(cta==0) g_p32.assign(sx.begin(),sx.begin()+32*SX);
        for(unsigned r=0;r<nr;++r){std::vector<u32> dg(2*SX,0);
            carry_seq(sx.data()+r*SX,dg.data(),NT*2,L+2);
            gpu31::reduce_sum<1>(dg.data(),L,bits,gap,q);
            for(unsigned cc=0;cc<L;++cc)out32[(r0+r)*L+cc]=dg[cc];}}

    // OC51 T2: nt0 outer, ks inner; acc[2 tiles][4 nt-slots][2 mt][2 q];
    // epilogue per tile over nt-slots -> carry/store -> next tile (ONE sX).
    std::vector<u32> out64(std::size_t(rows)*L,0);
    for(unsigned cta=0;cta*64<rows;++cta){const unsigned r0=cta*64,nr=(rows-r0<64)?rows-r0:64;
        long long acc[2][4][4][2][2][16][8]={};
        for(unsigned warp=0;warp<4;++warp)for(unsigned nt0=warp;nt0<NT;nt0+=8){
            unsigned ntA=nt0,ntB=nt0+4;bool hasB=ntB<NT;
            unsigned sl=(nt0-warp)/8;
            for(unsigned k=0;k<KS;++k){u32 B0[32][2],B1[32][2];
                for(int lane=0;lane<32;++lane){
                    B0[lane][0]=frag[((std::size_t(ntA)*KS+k)*32+lane)*2];
                    B0[lane][1]=frag[((std::size_t(ntA)*KS+k)*32+lane)*2+1];
                    if(hasB){B1[lane][0]=frag[((std::size_t(ntB)*KS+k)*32+lane)*2];
                        B1[lane][1]=frag[((std::size_t(ntB)*KS+k)*32+lane)*2+1];
                    }else B1[lane][0]=B1[lane][1]=0;}
                for(unsigned t=0;t<2;++t)for(unsigned mt=0;mt<2;++mt){u32 A[32][4];
                    for(int lane=0;lane<32;++lane){int g=lane>>2,tt=lane&3;
                        unsigned rr=r0+t*32+mt*16+g,rr8=rr+8;bool ok=rr<rows,ok8=rr8<rows;
                        A[lane][0]=ok?T[rr*SA+k*8+tt]:0;A[lane][1]=ok8?T[rr8*SA+k*8+tt]:0;
                        A[lane][2]=ok?T[rr*SA+k*8+4+tt]:0;A[lane][3]=ok8?T[rr8*SA+k*8+4+tt]:0;}
                    mma_emul(A,B0,acc[t][warp][sl][mt][0]);if(hasB)mma_emul(A,B1,acc[t][warp][sl][mt][1]);}}}
        for(unsigned t=0;t<2;++t){std::vector<u64> sx(std::size_t(32)*SX,0);
            for(unsigned warp=0;warp<4;++warp)for(unsigned sl=0;sl<4;++sl){
                unsigned nt0=warp+sl*8;if(nt0>=NT)break;
                unsigned ntA=nt0,ntB=nt0+4;bool hasB=ntB<NT;
                for(unsigned mt=0;mt<2;++mt){
                    epi_emul(acc[t][warp][sl][mt][0],ntA,SX,mt,sx.data());
                    if(hasB)epi_emul(acc[t][warp][sl][mt][1],ntB,SX,mt,sx.data());}}
            for(unsigned r=0;r<(nr<32?nr:32);++r){
                std::vector<u32> dg(2*SX,0);
                carry_seq(sx.data()+r*SX,dg.data(),NT*2,L+2);
                gpu31::reduce_sum<1>(dg.data(),L,bits,gap,q);
                if(r0+t*32+r<rows) for(unsigned cc=0;cc<L;++cc)out64[(r0+t*32+r)*L+cc]=dg[cc];}
        }}

    int bad32=0,bad64=0;std::vector<u32> ref(L);
    for(unsigned r=0;r<rows;++r){ref_row(T.data()+r*SA,W.data(),pitch,L,bits,gap,q,ref.data());
        for(unsigned cc=0;cc<L;++cc){if(out32[r*L+cc]!=ref[cc]){if(bad32<2)std::printf("BAD32 r=%u cc=%u got=%08x ref=%08x T0=%08x W=%08x\n",r,cc,out32[r*L+cc],ref[cc],T[r*SA+cc],W[cc*pitch+cc]);++bad32;}
            if(out64[r*L+cc]!=ref[cc]){if(bad64<2)std::printf("BAD64 r=%u cc=%u got=%08x ref=%08x\n",r,cc,out64[r*L+cc],ref[cc]);++bad64;}}}
    std::printf("DONE %-16s K=%3u rows=%3u bad32=%d bad64=%d\n",name,K,rows,bad32,bad64);std::fflush(stdout);
    return bad32+bad64;
}
int main(){std::printf("MAIN\n");std::fflush(stdout);std::mt19937_64 rng(0x0c51);int total=0;
    unsigned bits;u32 gap;std::vector<u32> q;
    Shape sh=mk(114,55);bits=32*sh.L-1;gap=4;q.assign(sh.L,0xffffffffu);q[0]=0xfffffffbu;
    const char* names[]={"random31","zeros","max-u32","max31+maxW","alt-huge","full-u32"};
    for(unsigned mode=0;mode<=5;++mode)for(unsigned rows:{64u,128u,100u})
        total+=run(sh,rows,mode,rng,bits,gap,q.data(),names[mode]);
    for(unsigned K:{8u,31u,63u}){Shape f=mk(K,55);
        total+=run(f,64,0,rng,bits,gap,q.data(),"fuzz");}
    std::printf("TOTAL BAD = %d\n",total);return total!=0;}
