"""Pinned unordered complex generalized Schur kernel and structural checks."""
import importlib.metadata
import math
import platform

from econforge.qz_admission import QZError, require


def runtime():
    versions = {p: importlib.metadata.version(p) for p in ("numpy","scipy","jax","jaxlib")}
    if versions != {"numpy":"2.2.6","scipy":"1.15.3","jax":"0.6.2","jaxlib":"0.6.2"} or platform.python_version() != "3.12.14":
        raise QZError("BACKEND_VERSION_MISMATCH","Pinned Python/numerical dependencies required",refused=True)
    import numpy as np
    import scipy
    from scipy import linalg
    require(np.__version__==versions["numpy"] and scipy.__version__==versions["scipy"],
            "BACKEND_VERSION_MISMATCH","Loaded backend differs from installed version")
    return np, linalg, versions


def frobenius(x):
    # hypot avoids squaring huge/tiny elements before summing.
    return math.hypot(*(float(v) for z in x.flat for v in (z.real,z.imag)))


def residual(np, x, y):
    d = max(frobenius(x),frobenius(y))
    require(math.isfinite(d), "DIAGNOSTIC_NONFINITE", "Unrepresentable Frobenius norm")
    return 0.0 if d==0 else frobenius(x/d-y/d)


def triangularity(np, x):
    d = frobenius(x)
    require(math.isfinite(d), "DIAGNOSTIC_NONFINITE", "Unrepresentable factor norm")
    return 0.0 if d==0 else frobenius(np.tril(x/d,-1))


def thresholds(np, A, B):
    n=A.shape[0]
    scale=max(frobenius(A),frobenius(B))
    if scale==0:
        raise QZError("ZERO_PENCIL","Exact all-zero A and B",refused=True)
    tau=1000*max(n,1)*float(np.finfo(np.float64).eps)
    pair=tau*scale
    require(math.isfinite(scale) and math.isfinite(pair) and pair>0,
            "THRESHOLD_UNREPRESENTABLE","Nonzero pencil threshold must be representable")
    return scale,tau,pair


def execute(linalg, A, B):
    return linalg.qz(B,A,output="complex",sort=None,overwrite_a=False,overwrite_b=False,check_finite=True)


def verify(np, A, B, output, tau):
    n=A.shape[0]
    require(type(output) in (tuple,list) and len(output)==4,
            "BACKEND_OUTPUT_INVALID","Four factors required")
    require(all(type(x) is np.ndarray and x.shape==(n,n) and x.dtype==np.dtype("complex128")
                and bool(np.isfinite(x).all()) for x in output),
            "BACKEND_OUTPUT_INVALID","Finite complex128 square backend arrays required")
    schur_B,schur_A,Q,Z=output
    with np.errstate(over="raise",invalid="raise",divide="raise"):
        d = {
            "reconstruction_B":residual(np,B,Q@schur_B@Z.conj().T),
            "reconstruction_A":residual(np,A,Q@schur_A@Z.conj().T),
            "unitarity_Q":frobenius(Q.conj().T@Q-np.eye(n))/math.sqrt(n),
            "unitarity_Z":frobenius(Z.conj().T@Z-np.eye(n))/math.sqrt(n),
            "triangularity_B":triangularity(np,schur_B),
            "triangularity_A":triangularity(np,schur_A)}
    require(all(math.isfinite(v) for v in d.values()),"DIAGNOSTIC_NONFINITE","Invalid structural diagnostics")
    return d


def pair_records(np, schur_B, schur_A, tau_pair):
    pairs=[]
    for alpha,beta in zip(np.diag(schur_B),np.diag(schur_A)):
        a,b=complex(alpha),complex(beta)
        value=None
        if abs(b)>tau_pair:
            status="FINITE"
            value=a/b
            require(math.isfinite(value.real) and math.isfinite(value.imag),
                    "SCALAR_NONFINITE","Admissible ratio must be finite")
        elif abs(a)>tau_pair:
            status="INFINITE"
        else:
            status="INDETERMINATE_0_OVER_0"
        pairs.append(dict(alpha=[a.real,a.imag],beta=[b.real,b.imag],
                          pair_status=status,lambda_value=None if value is None else [value.real,value.imag],
                          abs_alpha=abs(a),abs_beta=abs(b)))
    return pairs
