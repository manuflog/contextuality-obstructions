import numpy as np

def val2(x):
    if x==0: return 3
    v=0
    while x%2==0: x//=2; v+=1
    return v

UNIT_INV = {1:1,3:3,5:5,7:7}

def kernel_mod8(M):
    n, c = M.shape
    queue = [(M[i].astype(np.int64)%8, np.eye(1,n,i,dtype=np.int64)[0]) for i in range(n)]
    pivots = {}; kernel=[]; qi=0
    queue = list(queue)
    while qi < len(queue):
        row, tr = queue[qi]; qi+=1
        row = row.copy()%8; tr = tr.copy()%8
        while True:
            nz = np.nonzero(row)[0]
            if len(nz)==0:
                if tr.any(): kernel.append(tr%8)
                break
            j = nz[0]; a = int(row[j]); va = val2(a)
            if j in pivots:
                prow, ptr, vp = pivots[j]
                if va >= vp:
                    fac = (a // (1<<vp)) % 8
                    row = (row - fac*prow) % 8
                    tr  = (tr  - fac*ptr) % 8
                    continue
                else:
                    u = UNIT_INV[a >> va]
                    row = (u*row)%8; tr=(u*tr)%8
                    pivots[j] = (row, tr, va)
                    queue.append((prow, ptr))
                    if va>0: queue.append(((1<<(3-va))*row % 8, ((1<<(3-va))*tr)%8))
                    break
            else:
                u = UNIT_INV[a >> va]
                row=(u*row)%8; tr=(u*tr)%8
                pivots[j]=(row,tr,va)
                if va>0: queue.append(((1<<(3-va))*row%8, ((1<<(3-va))*tr)%8))
                break
    return kernel

def kernel_mod(M, d):
    assert d in (4,8)
    if d==8: return kernel_mod8(M)
    def val(x):
        if x==0: return 2
        return 0 if x%2 else 1
    UI={1:1,3:3}
    n,c=M.shape
    queue=[(M[i].astype(np.int64)%4, np.eye(1,n,i,dtype=np.int64)[0]) for i in range(n)]
    pivots={}; kernel=[]; qi=0
    while qi<len(queue):
        row,tr=queue[qi]; qi+=1
        row=row.copy()%4; tr=tr.copy()%4
        while True:
            nz=np.nonzero(row)[0]
            if len(nz)==0:
                if tr.any(): kernel.append(tr%4)
                break
            j=nz[0]; a=int(row[j]); va=val(a)
            if j in pivots:
                prow,ptr,vp=pivots[j]
                if va>=vp:
                    fac=(a>>vp)%4
                    row=(row-fac*prow)%4; tr=(tr-fac*ptr)%4; continue
                else:
                    u=UI[a>>va]; row=(u*row)%4; tr=(u*tr)%4
                    pivots[j]=(row,tr,va); queue.append((prow,ptr))
                    if va>0: queue.append(((1<<(2-va))*row%4, ((1<<(2-va))*tr)%4))
                    break
            else:
                u=UI[a>>va]; row=(u*row)%4; tr=(u*tr)%4
                pivots[j]=(row,tr,va)
                if va>0: queue.append(((1<<(2-va))*row%4,((1<<(2-va))*tr)%4))
                break
    return kernel

if __name__=="__main__":
    rng=np.random.default_rng(3)
    for d in (4,8):
        for trial in range(30):
            M=rng.integers(0,d,(5,4))
            K=kernel_mod(M,d)
            for t in K: assert not ((t@M)%d).any()
            from itertools import product as prod
            bf=[np.array(x) for x in prod(range(d),repeat=5) if not ((np.array(x)@M)%d).any() and any(x)]
            gen=set(); frontier=[np.zeros(5,dtype=int)]; gen.add(tuple(frontier[0]))
            while frontier:
                x=frontier.pop()
                for t in K:
                    y=tuple((np.array(x)+t)%d)
                    if y not in gen: gen.add(y); frontier.append(np.array(y))
            assert len(gen)==len(bf)+1, (d,trial,len(gen),len(bf)+1)
    print("Howell kernel unit tests (mod 4, mod 8): PASS - kernels exact vs brute force")
