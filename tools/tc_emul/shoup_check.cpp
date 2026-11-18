// Shoup product vs Montgomery product: must be bit-identical (both canonical).
#include <cstdint>
#include <cstdio>
#include <random>
typedef uint32_t u32; typedef uint64_t u64;
struct P{u32 p,ni,r2;};
u32 mont(u32 a,u32 b,P p){u64 t=u64(a)*b;u32 m=u32(t)*p.ni;u32 u=u32((t+u64(m)*p.p)>>32);return u>=p.p?u-p.p:u;}
u32 mulsh(u32 b,u32 w,u32 wq,u32 p){u32 q=u32((u64(b)*wq)>>32);u32 r=b*w-q*p;return r>=p?r-p:r;}
int main(){std::mt19937 g(3);int bad=0;for(u32 pp: {2147352577u,1073872897u,2013265921u}){u32 inv=1;for(int j=0;j<5;j++)inv*=2-pp*inv;u64 R=(u64(1)<<32)%pp;P p{pp,u32(0-inv),u32(R*R%pp)};
 for(int it=0;it<300000;it++){u32 wm=g()%pp;u32 w=mont(wm,1,p);u32 wq=u32((u64(w)<<32)/pp);u32 b=g();if(it&1)b%=pp;u32 a=mont(b%pp,wm,p);u32 c=mulsh(b%pp,w,wq,pp);if(a!=c)bad++;if(mont(w,p.r2,p)!=wm)bad++;}}
 printf("shoup bad=%d\n",bad);}
