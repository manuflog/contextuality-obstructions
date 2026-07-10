# V41 - TIER-2 ORBIT UNDER THE DERIVED AUTOMORPHISM GROUP (DROP=0).
# The cert4 family is the graph K33: 6 contexts (nodes), 9 observables (edges, each in
# exactly two contexts). Automorphisms fixing the deleted context = 12 graph autos, each
# dressed by a 64-torsor of spectral shifts t (spec(sigma v)=spec(v)+t_v, context sums
# matching s_{pi C}-s_C): 768 elements; with global conjugation, order 1536. The action
# on moment space is an exact monomial-rotation matrix (character indices through
# GL2(Z4), phases from t and s-substitution), so facets transport exactly - no snapping.
import numpy as np, itertools, json, os, pickle, time
from weyl import build
X,Z,w,tau,W,_=build(4,2)
fam=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
obs=sorted({t for C in fam for t in C})
def s_of(C):
    M=np.eye(16,dtype=complex)
    for v in C: M=M@W(np.array(v))
    return int(np.round(np.angle(np.trace(M)/abs(np.trace(M)))/(np.pi/2)))%4
S=[s_of(C) for C in fam]
spec={v:tuple(sorted(set(int(np.round(np.angle(x)/(np.pi/2)))%4 for x in np.linalg.eigvals(W(np.array(v)))))) for v in obs}
pair={v:frozenset(i for i,C in enumerate(fam) if v in C) for v in obs}
edge_of={pair[v]:v for v in obs}
if os.path.exists('/tmp/v41_gens.pkl'):
    gens=pickle.load(open('/tmp/v41_gens.pkl','rb'))
else:
    auts=[]
    for perm in itertools.permutations(range(1,6)):
        pi=(0,)+perm
        if all(frozenset(pi[i] for i in pair[v]) in edge_of for v in obs): auts.append(pi)
    def shifts_for(v,w_):
        a,b=spec[v],spec[w_]
        return [t for t in range(4) if tuple(sorted((x+t)%4 for x in a))==b]
    gens=[]
    for pi in auts:
        sig={v: edge_of[frozenset(pi[i] for i in pair[v])] for v in obs}
        if any(len(spec[v])!=len(spec[sig[v]]) for v in obs): continue
        choice={v:shifts_for(v,sig[v]) for v in obs}
        def bt(vs,t):
            if not vs:
                ok=all((sum(t[v] for v in fam[i]))%4==(S[pi[i]]-S[i])%4 for i in range(6))
                if ok: gens.append((pi,dict(sig),dict(t)))
                return
            v0=vs[0]
            for c in choice[v0]:
                t[v0]=c; bt(vs[1:],t); del t[v0]
        bt(list(obs),{})
    pickle.dump(gens,open('/tmp/v41_gens.pkl','wb'))
ids=sum(1 for pi,_,_ in gens if pi==tuple(range(6)))
print(f"group: {len(gens)} (sigma,t) elements = {ids} translations x {len(gens)//max(ids,1)} graph autos; x2 with conj")
CTX=[fam[i] for i in range(1,6)]
CH=[(a,b) for a in range(4) for b in range(4) if (a,b)!=(0,0)]
CHi={ab:k for k,ab in enumerate(CH)}
dd=np.load('/tmp/v35_cache.npz'); V=dd['V']
gpair=[(C[0],C[1]) for C in CTX]; third=[C[2] for C in CTX]
evec={}
for j,C in enumerate(CTX):
    evec[(j,C[0])]=(1,0); evec[(j,C[1])]=(0,1); evec[(j,C[2])]=(-1,-1)
rem=list(range(1,6)); pos={c:k for k,c in enumerate(rem)}
def build_maps(gen):
    pi,sig,t=gen
    inv={sig[v]:v for v in sig}
    maps=[]
    for i in range(5):
        ci=rem[i]; cj=[c for c in rem if pi[c]==ci][0]; jj=pos[cj]
        h1,h2=gpair[i]; u1,u2=inv[h1],inv[h2]
        e1,e2=evec[(jj,u1)],evec[(jj,u2)]
        row=[]
        for (a,b) in CH:
            ap=(a*e1[0]+b*e2[0])%4; bp=(a*e1[1]+b*e2[1])%4
            const=(a*t[u1]+b*t[u2]+(a*(u1==third[jj])+b*(u2==third[jj]))*S[cj])%4
            row.append((jj,CHi[(ap,bp)],const))
        maps.append(row)
    return maps
def apply_gen(vec,maps):
    out=np.zeros(150)
    for i in range(5):
        for k,(jj,kk,const) in enumerate(maps[i]):
            th=-const*np.pi/2; c,s_=np.cos(th),np.sin(th)
            re,im=vec[(jj*15+kk)*2],vec[(jj*15+kk)*2+1]
            out[(i*15+k)*2]=c*re-s_*im; out[(i*15+k)*2+1]=s_*re+c*im
    return out
s2g=2*np.sqrt(2)
def snap2(f):
    g1=np.round(f*4)/4; g2=np.round(f*s2g)/s2g
    return np.where(np.abs(f-g1)<=np.abs(f-g2),g1,g2)
def key(f):
    g=snap2(f)
    out=[]
    for x in g:
        q=x*4
        if abs(q-round(q))<1e-9: out.append(int(round(q)))
        else: out.append(10000+int(round(x*s2g)))
    return tuple(out)
Vkeys={key(v) for v in V}
trans=[g for g in gens if g[0]==tuple(range(6))]
span={tuple([0]*9)}; tsel=[]
for g in trans:
    tv=tuple(g[2][v] for v in obs)
    if tv not in span:
        tsel.append(g)
        span={tuple((np.array(a)+k*np.array(tv))%4) for a in span for k in range(4)}
    if len(span)==64: break
pis=sorted({g[0] for g in gens})
def compose(p,q): return tuple(p[q[i]] for i in range(6))
def closure(gsel):
    got={tuple(range(6))}; fr=list(gsel)
    while fr:
        p=fr.pop()
        for q in list(got):
            for r in (compose(p,q),compose(q,p)):
                if r not in got: got.add(r); fr.append(r)
        got.add(p)
    return got
gsel=[]
for p in pis:
    if p==tuple(range(6)): continue
    if len(closure(gsel+[p]))>len(closure(gsel)): gsel.append(p)
    if len(closure(gsel))==len(pis): break
reps=[next(g for g in gens if g[0]==p) for p in gsel]
GEN=tsel+reps
MAPS=[build_maps(g) for g in GEN]
print(f"generating set: {len(tsel)} translations + {len(reps)} graph reps (+conj); pi-group order {len(pis)}")
ok=True
for M in MAPS:
    imgs={key(apply_gen(v,M)) for v in V}
    ok &= (imgs==Vkeys)
print(f"action correctness (each generator BIJECTS the 64 vertices): {ok}")
f2=np.load('/tmp/v36_facets.npz')
seen={}; frontier=[]
for f0,b0 in zip(f2['F'],f2['B']):
    kk=key(f0)
    if kk not in seen: seen[kk]=(snap2(f0),int(b0)); frontier.append((snap2(f0),int(b0)))
t0=time.time(); partial=False
while frontier:
    if time.time()-t0>700: partial=True; break
    f,b=frontier.pop()
    cands=[apply_gen(f,M) for M in MAPS]
    fc=f.copy(); fc[1::2]*=-1; cands.append(fc)
    for f2c in cands:
        k2=key(f2c)
        if k2 not in seen:
            fs=snap2(f2c); seen[k2]=(fs,b); frontier.append((fs,b))
print(f"tier-2 facet orbit: {len(seen)} facets (partial={partial}, t={time.time()-t0:.0f}s)")
FO=np.array([v[0] for v in seen.values()]); BO=np.array([v[1] for v in seen.values()],float)
bad=float((np.max(V@FO.T,axis=0)-BO).max())
print(f"validity over orbit: max(NCmax - b) = {bad:.2e}")
np.savez('/tmp/v41_orbit.npz',F=FO,B=BO,partial=partial)
print("PASS" if ok and bad<1e-7 and not partial else ("PARTIAL" if partial else "FAIL"))
