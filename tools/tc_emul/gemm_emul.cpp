#include "tc.cuh"
#include <cstdio>
#include <random>
using namespace tc;
// emulate one warp mma with documented layout; returns C[16][8]
static void mma_emul(const u32 A[32][4], const u32 Bf[32][2], long long C[16][8]){
  unsigned char a[16][32]={}, b[32][8]={};
  for(int lane=0;lane<32;lane++){int g=lane>>2,t=lane&3;
    for(int i=0;i<4;i++){
      a[g][t*4+i]=A[lane][0]>>(8*i); a[g+8][t*4+i]=A[lane][1]>>(8*i);
      a[g][16+t*4+i]=A[lane][2]>>(8*i); a[g+8][16+t*4+i]=A[lane][3]>>(8*i);
      b[t*4+i][g]=Bf[lane][0]>>(8*i); b[16+t*4+i][g]=Bf[lane][1]>>(8*i);}}
  for(int r=0;r<16;r++)for(int c=0;c<8;c++)for(int k=0;k<32;k++)C[r][c]+=a[r][k]*b[k][c];
}
int main(){
  std::mt19937_64 rng(1);
  // ---- encode: L=55, K=118 random primes-ish (any odd < 2^31 > 2^30)
  unsigned L=55,K=118; std::vector<u32> ps; for(unsigned k=0;k<K;k++) ps.push_back((1u<<30)+ 2*(rng()%(1u<<29))+1);
  unsigned KS,NT; auto f=bake_encode(ps,L,KS,NT);
  std::vector<u32> X(16*56,0); for(unsigned r=0;r<16;r++)for(unsigned c=0;c<L;c++)X[r*56+c]=u32(rng());
  int bad=0;
  for(unsigned nt=0;nt<NT;nt++){ long long C[16][8]={};
    for(unsigned ks=0;ks<KS;ks++){u32 A[32][4],B[32][2];
      for(int lane=0;lane<32;lane++){int g=lane>>2,t=lane&3;
        A[lane][0]=X[g*56+ks*8+t];A[lane][1]=X[(g+8)*56+ks*8+t];A[lane][2]=X[g*56+ks*8+4+t];A[lane][3]=X[(g+8)*56+ks*8+4+t];
        B[lane][0]=f[((nt*KS+ks)*32+lane)*2];B[lane][1]=f[((nt*KS+ks)*32+lane)*2+1];}
      mma_emul(A,B,C);}
    // epilogue emulate
    u64 tot[32];
    for(int lane=0;lane<32;lane++){int t=lane&3,g=lane>>2; unsigned sh=(t&1)*16;
      u64 pg=(u64(u32(C[g][2*t]))<<sh)+(u64(u32(C[g][2*t+1]))<<(sh+8));
      u64 pg8=(u64(u32(C[g+8][2*t]))<<sh)+(u64(u32(C[g+8][2*t+1]))<<(sh+8));
      int pl=lane^1; int tp=pl&3; unsigned shp=(tp&1)*16;
      u64 ppg=(u64(u32(C[g][2*tp]))<<shp)+(u64(u32(C[g][2*tp+1]))<<(shp+8));
      u64 ppg8=(u64(u32(C[g+8][2*tp]))<<shp)+(u64(u32(C[g+8][2*tp+1]))<<(shp+8));
      bool odd=t&1; tot[lane]=(odd?pg8:pg)+(odd?ppg8:ppg);
      unsigned k=nt*2+(t>>1), row=g+(odd?8:0); if(k>=K)continue;
      u64 p=ps[k]; // want X*2^32 mod p
      unsigned __int128 acc=0; for(int c=L-1;c>=0;c--) acc=((acc<<32)+X[row*56+c])%p; acc=(acc<<32)%p;
      if(tot[lane]%p!=(u64)acc) bad++;}
  }
  printf("encode bad=%d KS=%u NT=%u\n",bad,KS,NT);
  // ---- crt: W random L x pitch, terms random 31-bit
  unsigned pitch=119; std::vector<u32> W(L*pitch); for(auto&w:W)w=u32(rng());
  unsigned cKS,cNT; auto g2=bake_crt(W,L,pitch,cKS,cNT); unsigned SA=crt_sa_stride(cKS);
  std::vector<u32> T(16*SA,0); for(unsigned r=0;r<16;r++)for(unsigned j=0;j<pitch;j++)T[r*SA+j]=u32(rng())&0x7fffffff;
  std::vector<u64> sX(16*(2*cNT),0);
  for(unsigned nt=0;nt<cNT;nt++){ long long C[16][8]={};
    for(unsigned ks=0;ks<cKS;ks++){u32 A[32][4],B[32][2];
      for(int lane=0;lane<32;lane++){int g=lane>>2,t=lane&3;
        A[lane][0]=T[g*SA+ks*8+t];A[lane][1]=T[(g+8)*SA+ks*8+t];A[lane][2]=T[g*SA+ks*8+4+t];A[lane][3]=T[(g+8)*SA+ks*8+4+t];
        B[lane][0]=g2[((nt*cKS+ks)*32+lane)*2];B[lane][1]=g2[((nt*cKS+ks)*32+lane)*2+1];}
      mma_emul(A,B,C);}
    for(int lane=0;lane<32;lane++){int t=lane&3,g=lane>>2; unsigned sh=(t&1)*16; bool odd=t&1;
      int tp=(lane^1)&3; unsigned shp=(tp&1)*16;
      u64 pg=(u64(u32(C[g][2*t]))<<sh)+(u64(u32(C[g][2*t+1]))<<(sh+8));
      u64 pg8=(u64(u32(C[g+8][2*t]))<<sh)+(u64(u32(C[g+8][2*t+1]))<<(sh+8));
      u64 ppg=(u64(u32(C[g][2*tp]))<<shp)+(u64(u32(C[g][2*tp+1]))<<(shp+8));
      u64 ppg8=(u64(u32(C[g+8][2*tp]))<<shp)+(u64(u32(C[g+8][2*tp+1]))<<(shp+8));
      sX[(g+(odd?8:0))*(2*cNT)+nt*2+(t>>1)]=(odd?pg8:pg)+(odd?ppg8:ppg);}
  }
  int cbad=0;
  for(unsigned r=0;r<16;r++){ std::vector<u32> d(L+2,0),ref(L+2,0); u64 c=0;
    for(unsigned w=0;w<2*cNT;w++){c+=sX[r*2*cNT+w];d[w]=u32(c);c>>=32;} for(unsigned w=2*cNT;w<L+2;w++){d[w]=u32(c);c>>=32;}
    // ref: sum_j T_j * W_j
    for(unsigned j=0;j<pitch;j++){u64 cc=0; for(unsigned i=0;i<L+2;i++){u64 v=cc+ref[i]+(i<L?u64(T[r*SA+j])*W[i*pitch+j]:0); ref[i]=u32(v); cc=v>>32;}}
    if(d!=ref)cbad++;}
  printf("crt bad rows=%d KS=%u NT=%u SA=%u\n",cbad,cKS,cNT,SA);
}
