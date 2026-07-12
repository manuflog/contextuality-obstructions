# V38 - OBSERVER-CONTEXT CATEGORY (switch subgroupoid + non-invertible refinements):
#        the two machine checks Paper D needs. [file name kept for stability.]
# (A) PULLBACK MONOTONICITY: the Wigner-enlargement (adjoin a friend qubit, embed the
#     system family as A (x) I_F) is a strict morphism of context groupoids pulling the
#     cocycle back; the PM obstruction survives verbatim at d=8: any global section of
#     the enlarged observer family would restrict to one for PM. Machine: the six
#     embedded PM context products are (+1 x5, -1 x1), identical to PM.
# (B) COMPLEMENTARITY INVARIANT: on the Weyl section, hol(loop) = det of the switching
#     product against an SU closer; hol itself is section-dependent, but hol^2 in {+-1}
#     is invariant under Weyl regauging of every switch. Machine: MUB triangle
#     (T -> Tx -> Ty -> T) gives hol^2 = -1 under ALL random Weyl regaugings and SU
#     closers; loops whose switches stabilize a single context give hol^2 = +1.
#     (V12 pins hol = -i in the Clifford presentation and hardware-excludes -i at ray
#     level ~56 sigma; here the gauge-robust Z2 invariant is isolated.)
# ADVERSARIAL PASS #1 PINS (Paper D corrections):
#   * hol^2 invariance is PROVED, not just sampled: every Weyl element has det +-1 and
#     every closer is SU, so regauging multiplies hol by +-1 only. Machine-checked below.
#   * CONVERSE FAILS: the 2-cycle T -> Tx -> T with switches (H,H) traverses a
#     complementary pair yet has hol^2 = +1 (det H^2 = 1). So hol^2 = -1 detects the
#     ORIENTED MUB TRIANGLE (odd tau-switch parity), NOT pairwise complementarity;
#     nontriviality REQUIRES complementary switching but not conversely.
#   * scope: the +1 statement holds for WEYL switches; non-Weyl stabilizer elements
#     (generic diagonals) have arbitrary det and are excluded by the Weyl-gauge setup.
import numpy as np, itertools
rng=np.random.default_rng(11)
I2=np.eye(2); H=np.array([[1,1],[1,-1]])/np.sqrt(2); S=np.diag([1,1j])
X=np.array([[0,1],[1,0]]); Z=np.diag([1,-1]); Y=1j*X@Z
P1={'I':I2,'X':X,'Y':Y,'Z':Z}
def kron(*Ms):
    out=Ms[0]
    for M in Ms[1:]: out=np.kron(out,M)
    return out
# ---- (A) ----
def pm_op(s): return kron(P1[s[0]],P1[s[1]])
ROWS=[['XI','IX','XX'],['IZ','ZI','ZZ'],['XZ','ZX','YY']]
COLS=[['XI','IZ','XZ'],['IX','ZI','ZX'],['XX','ZZ','YY']]
prods=[]
for C in ROWS+COLS:
    M=np.eye(8,dtype=complex)
    for s in C: M=M@kron(pm_op(s),I2)      # embed with friend qubit
    prods.append(int(np.round(np.real(np.trace(M))/8)))
okA=sorted(prods)==[-1,1,1,1,1,1]
print(f"(A) embedded PM (x) I_F products at d=8: {prods} -> obstruction survives: {okA}")
# ---- (B) ----
WEYL=[ (1j**k)*P1[p] for k in range(4) for p in 'IXYZ' ]   # Weyl-section gauge group
def su_closer(Uacc):
    th=rng.uniform(0,2*np.pi); D=np.diag([np.exp(1j*th),np.exp(-1j*th)])
    P=np.eye(2) if rng.integers(2)==0 else np.array([[0,1],[-1,0]])
    Wc=D@P@np.linalg.inv(Uacc); Wc/=np.linalg.det(Wc)**0.5
    return Wc
def hol2(switches,ntrials=200):
    vals=set()
    for _ in range(ntrials):
        Us=[rng.choice(WEYL)@U@rng.choice(WEYL) for U in switches]
        acc=np.eye(2,dtype=complex)
        for U in Us: acc=U@acc
        L=su_closer(acc)@acc
        h=np.linalg.det(L)
        vals.add(complex(np.round(h*h,6)))
    return vals
v_mub=hol2([H,S])                 # T -> Tx -> Ty -> T (complementary MUB triangle)
stab_loops=[[Z,Z],[X,Z],[Z],[X,X,Z,Z],[X]]   # switches inside one context's Weyl stabilizer
v_stab=set()
for sw in stab_loops: v_stab|=hol2(sw,80)
v_2cyc=hol2([H,H],120)   # complementary 2-cycle: converse-failure certificate
dets_ok=all(abs(abs(np.linalg.det(Wg))-1)<1e-9 and abs(np.linalg.det(Wg).imag)<1e-9
            or abs(np.linalg.det(Wg).real)<1e-9 for Wg in WEYL)
dets_pm1=all(abs(np.linalg.det(Wg)**2-1)<1e-9 for Wg in WEYL)
okB=(v_mub=={complex(-1)}) and (v_stab=={complex(1)}) and (v_2cyc=={complex(1)}) and dets_pm1
print(f"(B) hol^2: MUB triangle {sorted(v_mub,key=str)} | Weyl within-context loops {sorted(v_stab,key=str)}")
print(f"(B) CONVERSE-FAILURE certificate: complementary 2-cycle (H,H) hol^2 = {sorted(v_2cyc,key=str)} (= +1)")
print(f"(B) proof ingredient machine-checked: det(w)^2 = 1 for all 16 Weyl elements: {dets_pm1}")
print("PASS" if okA and okB else "FAIL")
