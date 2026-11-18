#include <cstdint>
#include <cstdio>
#include <random>
typedef uint32_t u32; typedef uint64_t u64;
struct P{u32 p,ni;};
u32 mont(u32 a,u32 b,P p){u64 t=u64(a)*b;u32 m=u32(t)*p.ni;u32 u=u32((t+u64(m)*p.p)>>32);return u>=p.p?u-p.p:u;}
int main(){std::mt19937 g(5);u32 pp=2013265921u; u32 inv=1;for(int j=0;j<5;j++)inv*=2-pp*inv;P p{pp,u32(0-inv)};int bad=0;
for(int it=0;it<100000;it++){u32 r=g()%pp,u=g()%pp,iv=g()%pp;u32 dec=mont(mont(r,u,p),1,p);u32 want=u32(u64(dec)*iv%pp);u32 got=mont(r,mont(u,iv,p),p);if(got!=want)bad++;}printf("bad=%d\n",bad);}
