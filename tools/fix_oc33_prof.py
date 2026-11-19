p='tc.cuh'
s=open(p,encoding='utf-8').read()
add='    TCP_ADD(8, f3, f4); TCP_ADD(9, f4, f5);\n'
# insert inside both branches right after TCP_ADD(5..7)
pat='    TCP_ADD(5, f0, f1); TCP_ADD(6, f1, f2); TCP_ADD(7, f2, f3);\n'
assert s.count(pat)==2, s.count(pat)
s=s.replace(pat, pat+add)
# remove the old out-of-scope copy
old=add+add+'}'
# after replacement there are three copies: two inside, one outside; find the one followed by '}'
i=s.find(add, s.find(pat, s.find(pat)+1)+len(pat)+len(add))
# simpler: locate '#endif\n' + add + '}'
outside='#endif\n'+add+'}'
assert s.count(outside)==1
s=s.replace(outside, '#endif\n}')
open(p,'w',encoding='utf-8').write(s)
print('fixed profiling scope')
