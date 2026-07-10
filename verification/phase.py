# make_alg is imported by spectrum_test2 but never called there (dead import).
# Provide a harmless stub so the module import succeeds. The load-bearing phase
# algebra (make_alg2 / phexp) is defined in spectrum_test2 itself and is
# cross-validated against weyl.build's matrices by that file's check A.
def make_alg(*a, **k):
    raise NotImplementedError("phase.make_alg stub: not used by the pipeline")
