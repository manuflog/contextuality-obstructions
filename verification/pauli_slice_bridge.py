# V11 - Two independent routes to the PM obstruction agree:
#   carry/self-pairing route (Paper B machinery, criterion.py): unique cycle, Q=1
#   matrix route (pinned Weyl convention): product of six context signs = -1
# Machine-checks Paper A Thm E/F (previously analytic-only in INDEX). Expected: AGREE.
import numpy as np
# Imported helper modules below print their own verdicts at import time. Their output is
# suppressed here so that this script's stdout contains ONLY this script's verdict.
# Reason (2026-07-27): run_all.sh judges a script by grepping for a verdict token. When an
# imported module printed its own PASS into this script's stdout, that gate could be satisfied
# by a token belonging to a different process -- which is exactly how nine dead scripts passed.
# A verdict is only evidence if it is attributable to the thing being judged.
import contextlib as _ctx, io as _io
with _ctx.redirect_stdout(_io.StringIO()):
    from criterion import carry_data, left_kernel, sympinv
from weyl import build
XI=(1,0,0,0);IX=(0,0,1,0);XX=(1,0,1,0);IZ=(0,0,0,1);ZI=(0,1,0,0)
ZZ=(0,1,0,1);XZ=(1,0,0,1);ZX=(0,1,1,0);YY=(1,1,1,1)
fam=[[XI,IX,XX],[IZ,ZI,ZZ],[XZ,ZX,YY],[XI,IZ,XZ],[IX,ZI,ZX],[XX,ZZ,YY]]
A,b=carry_data(fam,2); K=left_kernel(A)
assert len(K)==1 and all(K[0]==1), "PM must have the unique all-ones cycle"
Q=sympinv(fam,K[0],2)
X,Z,w,tau,Wf,N=build(2,2)
sign=1.0
for C in fam:
    M=np.eye(4,dtype=complex)
    for v in C: M=M@Wf(np.array(v))
    sign*= np.trace(M).real/4
print(f"carry route: unique cycle, Q = {Q} (odd => obstruction present)")
print(f"matrix route: six-sign product = {round(sign)}")
print("PASS: routes AGREE" if (Q==1 and round(sign)==-1) else "FAIL")
