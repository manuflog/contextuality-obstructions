# Minimal reconstruction of the Weyl operator builder.
# Convention PINNED by verify_cert8.py and verify_cert16.py (both self-contained):
#   W(v) = tau^{-sum_i a_i b_i} (x)_i X^{a_i} Z^{b_i},  tau=e^{i pi/d}, w=tau^2.
# build(d,m) returns a 6-tuple; spectrum_test2 uses slots [2],[3],[4] = w,tau,W.
import numpy as np
def build(d,m):
    w=np.exp(2j*np.pi/d); tau=np.exp(1j*np.pi/d)
    X=np.roll(np.eye(d),1,axis=0); Z=np.diag([w**k for k in range(d)])
    Xp=[np.linalg.matrix_power(X,a) for a in range(d)]
    Zp=[np.linalg.matrix_power(Z,b) for b in range(d)]
    def T1(a,b): return Xp[a%d]@Zp[b%d]
    def W(v):
        M=T1(v[0],v[1])
        for i in range(1,m): M=np.kron(M,T1(v[2*i],v[2*i+1]))
        q=sum(int(v[2*i])*int(v[2*i+1]) for i in range(m))
        return tau**(-q)*M
    return (X,Z,w,tau,W,d**m)
