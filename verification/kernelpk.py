import numpy as np

def kernel_modpk(M, p, k):
    n = p**k
    def val(x):
        if x==0: return k
        v=0
        while x%p==0: x//=p; v+=1
        return v
    nr, nc = M.shape
    queue=[(M[i].astype(np.int64)%n, np.eye(1,nr,i,dtype=np.int64)[0]) for i in range(nr)]
    pivots={}; kernel=[]; qi=0
    while qi<len(queue):
        row,tr=queue[qi]; qi+=1
        row=row.copy()%n; tr=tr.copy()%n
        while True:
            nz=np.nonzero(row)[0]
            if len(nz)==0:
                if tr.any(): kernel.append(tr%n)
                break
            j=int(nz[0]); a=int(row[j]); va=val(a)
            if j in pivots:
                prow,ptr,vp=pivots[j]
                if va>=vp:
                    fac=(a//(p**vp)) % n
                    row=(row-fac*prow)%n; tr=(tr-fac*ptr)%n; continue
                else:
                    u=pow(a//(p**va), -1, n)
                    row=(u*row)%n; tr=(u*tr)%n
                    pivots[j]=(row,tr,va); queue.append((prow,ptr))
                    if va>0: queue.append(((p**(k-va))*row%n,((p**(k-va))*tr)%n))
                    break
            else:
                u=pow(a//(p**va), -1, n)
                row=(u*row)%n; tr=(u*tr)%n
                pivots[j]=(row,tr,va)
                if va>0: queue.append(((p**(k-va))*row%n,((p**(k-va))*tr)%n))
                break
    return kernel

if __name__=="__main__":
    from itertools import product as prod
    rng=np.random.default_rng(11)
    for (p,k) in [(2,4),(2,2),(3,1),(3,2),(5,1)]:
        n=p**k
        for trial in range(25):
            M=rng.integers(0,n,(5,4))
            K=kernel_modpk(M,p,k)
            for t in K: assert not ((t@M)%n).any()
            bf=[x for x in prod(range(n),repeat=5) if not ((np.array(x)@M)%n).any()]
            gen={(0,)*5}; frontier=[np.zeros(5,dtype=int)]
            while frontier:
                x=frontier.pop()
                for t in K:
                    y=tuple((x+t)%n)
                    if y not in gen: gen.add(y); frontier.append(np.array(y))
            assert len(gen)==len(bf), (p,k,trial,len(gen),len(bf))
        print(f"kernel_modpk unit test p={p},k={k} (n={n}): PASS exact vs brute force")
