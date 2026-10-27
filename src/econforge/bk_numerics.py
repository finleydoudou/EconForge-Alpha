"""P3.7 frozen numerical conditions only; no graph or solution recovery."""
import math

from econforge.qz_admission import QZError, require
from econforge import qz_backend

ROOT_BAND = 1e-6
PAIR_ZERO = 1e-12
ROOT_MATCH = 1e-8


class BKError(QZError):
    def __init__(self, code, message, *, refused=False, bk_status="NOT_EVALUATED"):
        super().__init__(code, message, refused=refused)
        self.bk_status = bk_status


def unit_class(alpha, beta):
    """Homogeneous closed-band classification; separate pair admission precedes it."""
    a, b = abs(complex(alpha)), abs(complex(beta))
    require(math.isfinite(a) and math.isfinite(b), "PAIR_NONFINITE", "Finite homogeneous moduli required")
    if a == b == 0:
        return "UNRESOLVED_PAIR"
    # Most inputs need no rescaling; this preserves the exact binary64 endpoints.
    if b > float.fromhex('0x1.fffffffffffffp+1023') / (1 + ROOT_BAND):
        a, b = a / 2, b / 2
    if a < (1 - ROOT_BAND) * b:
        return "STABLE"
    if a > (1 + ROOT_BAND) * b:
        return "OUTSIDE"
    return "BOUNDARY"


def pair_scale(np, A, B):
    """Section-9 infinity scale, computed safely and never used to replace A/B."""
    largest = float(max(np.max(np.abs(A)), np.max(np.abs(B))))
    require(largest > 0 and math.isfinite(largest), "ZERO_PENCIL", "Nonzero finite pencil required")
    rows = max(float(np.max(np.sum(np.abs(A / largest), axis=1))),
               float(np.max(np.sum(np.abs(B / largest), axis=1))))
    return largest, rows


def classify_pairs(np, A, B, alpha, beta):
    largest, rows = pair_scale(np, A, B)
    classes, small_denominators = [], []
    for i, (a, b) in enumerate(zip(alpha, beta)):
        aa, bb = abs(complex(a)) / largest / rows, abs(complex(b)) / largest / rows
        require(math.isfinite(aa) and math.isfinite(bb), "PAIR_NONFINITE", "Pair normalization not representable")
        classes.append("UNRESOLVED_PAIR" if max(aa, bb) <= PAIR_ZERO else unit_class(a, b))
        if bb <= PAIR_ZERO and b != 0:
            small_denominators.append(i)
    return classes, small_denominators


def singular_margin(np, linalg, matrix):
    values = linalg.svdvals(matrix, check_finite=True)
    require(type(values) is np.ndarray and values.ndim == 1 and len(values) > 0
            and bool(np.isfinite(values).all()) and bool((values >= 0).all()),
            "SVD_OUTPUT_INVALID", "Finite nonnegative singular values required")
    hi, lo = float(np.max(values)), float(np.min(values))
    margin = 0.0 if hi == 0 else lo / hi
    require(math.isfinite(margin), "SVD_OUTPUT_INVALID", "Finite relative margin required")
    return dict(sigma_max=hi, sigma_min=lo, margin=margin)


def regularity(np, linalg, A, B):
    n = len(A)
    scale = float(max(np.max(np.abs(A)), np.max(np.abs(B))))
    if scale == 0:
        raise BKError("ZERO_PENCIL", "Exact all-zero pencil", refused=True)
    tau = 1000 * max(n, 1) * float(np.finfo(np.float64).eps)
    As, Bs = A / scale, B / scale
    witnesses, winner = [], None
    for mu in range(n + 1):
        d = singular_margin(np, linalg, Bs - mu * As)
        item = dict(mu=mu, rho=d['margin'], sigma_min=d['sigma_min'], sigma_max=d['sigma_max'])
        witnesses.append(item)
        if winner is None and item['rho'] > tau:
            winner = item
    return dict(status="PASS" if winner else "UNRESOLVED", common_scale=scale,
                tau_reg=tau, witness_mu=None if winner is None else winner['mu'],
                witness_rho=None if winner is None else winner['rho'], witnesses=witnesses,
                claim="SUFFICIENT_NUMERICAL_ADMISSION_NOT_SYMBOLIC_PROOF")


def projection(np, linalg, Z, state_rows, stable_count):
    ns = len(state_rows)
    tau = 1000 * max(ns, 1) * float(np.finfo(np.float64).eps)
    require(len(set(state_rows)) == ns and all(type(i) is int and 0 <= i < len(Z) for i in state_rows),
            "PARTITION_MISMATCH", "Explicit distinct state rows required")
    out = dict(status="NOT_RUN", projection_margin=None, tau_projection=tau,
               state_row_indices=list(state_rows), sigma_min=None, sigma_max=None)
    if stable_count != ns:
        return out
    if ns == 0:
        out.update(status="PASS", vacuous=True)
        return out
    M = Z[state_rows, :ns]
    require(M.shape == (ns, ns), "PROJECTION_SHAPE", "Square state projection required")
    d = singular_margin(np, linalg, M)
    out.update(status="PASS" if d['margin'] > tau else "UNRESOLVED",
               projection_margin=d['margin'], sigma_min=d['sigma_min'], sigma_max=d['sigma_max'], vacuous=False)
    return out


def normalized_pairs(pairs):
    out = []
    for a, b in pairs:
        a, b = complex(a), complex(b)
        scale = max(abs(a), abs(b))
        require(scale > 0 and math.isfinite(scale), "UNRESOLVED_PAIR", "Resolved pairs required for matching")
        a, b = a / scale, b / scale
        norm = math.hypot(abs(a), abs(b))
        out.append((a / norm, b / norm))
    return out


def spectrum_match(left, right):
    """Section-11 one-to-one minimum-maximum projective matching, no test imports."""
    require(len(left) == len(right) and len(left) > 0, "SPECTRUM_SIZE", "Identical nonempty multiplicity required")
    p, q = normalized_pairs(left), normalized_pairs(right)
    distances = [[abs(a * d - b * c) for c, d in q] for a, b in p]
    levels = sorted({d for row in distances for d in row})

    def assignment(threshold):
        matched = [-1] * len(q)
        def augment(i, seen):
            for j, value in enumerate(distances[i]):
                if value <= threshold and j not in seen:
                    seen.add(j)
                    if matched[j] == -1 or augment(matched[j], seen):
                        matched[j] = i
                        return True
            return False
        if not all(augment(i, set()) for i in range(len(p))):
            return None
        result = [None] * len(p)
        for j, i in enumerate(matched):
            result[i] = j
        return result

    lo, hi = 0, len(levels) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if assignment(levels[mid]) is None:
            lo = mid + 1
        else:
            hi = mid
    mapping = assignment(levels[lo])
    return dict(passed=levels[lo] <= ROOT_MATCH, max_projective_distance=levels[lo],
                threshold=ROOT_MATCH, matching=mapping)


def conjugacy(pairs, classes):
    result = spectrum_match(pairs, [(complex(a).conjugate(), complex(b).conjugate()) for a, b in pairs])
    require(result['passed'], "CONJUGACY_UNRESOLVED", "Real-system conjugacy not resolved")
    require(all(classes[i] == classes[j] for i, j in enumerate(result['matching'])),
            "CONJUGACY_CLASS_MISMATCH", "Conjugate classes cannot split")
    return result


def order(np, linalg, A, B):
    def select(alpha, beta):
        shape = np.asarray(alpha).shape
        labels = [unit_class(a, b) for a, b in zip(np.asarray(alpha).flat, np.asarray(beta).flat)]
        require(all(c in ('STABLE', 'OUTSIDE') for c in labels), "ORDER_SELECTOR_UNRESOLVED", "Unexpected unresolved backend pair")
        return np.asarray([c == 'STABLE' for c in labels], dtype=bool).reshape(shape)
    return linalg.ordqz(B, A, sort=select, output="complex", overwrite_a=False, overwrite_b=False, check_finite=True)


def verify_order(np, A, B, output, old_pairs, old_classes, tau):
    require(type(output) in (tuple, list) and len(output) == 6, "ORDERED_OUTPUT_INVALID", "Six ordqz outputs required")
    SB, SA, alpha, beta, Q, Z = output
    diagnostics = qz_backend.verify(np, A, B, (SB, SA, Q, Z), tau)
    if any(v > tau for v in diagnostics.values()):
        error = BKError('ORDERED_QZ_VERIFICATION_FAILED', 'Ordered structural checks failed')
        error.diagnostics = dict(structural=diagnostics, tau_qz=tau, status='FAIL')
        raise error
    n = len(A)
    require(all(type(x) is np.ndarray and x.shape == (n,) and x.dtype == np.dtype('complex128')
                and bool(np.isfinite(x).all()) for x in (alpha, beta)), "ORDERED_PAIR_INVALID", "Finite complex128 diagonal pairs required")
    require(bool(np.array_equal(alpha, np.diag(SB))) and bool(np.array_equal(beta, np.diag(SA))),
            "ORDERED_DIAGONAL_MISMATCH", "Returned pairs must be factor diagonals")
    classes, small = classify_pairs(np, A, B, alpha, beta)
    stable_count = old_classes.count('STABLE')
    require(classes == ['STABLE'] * stable_count + ['OUTSIDE'] * (n - stable_count),
            "ORDERED_CLASS_MISMATCH", "Reclassified roots/count/contiguity differ")
    pairs = list(zip(alpha, beta))
    matched = spectrum_match(old_pairs, pairs)
    require(matched['passed'], "ORDERED_SPECTRUM_MISMATCH", "Supplied spectrum not preserved")
    require(all(old_classes[i] == classes[j] for i, j in enumerate(matched['matching'])),
            "ORDERED_SPECTRUM_CLASS_MISMATCH", "Matched roots changed class")
    conj = conjugacy(pairs, classes)
    return dict(structural=diagnostics, spectrum=matched, conjugacy=conj,
                stable_count=stable_count, stable_block_contiguous=True,
                classes=classes, small_nonzero_beta_indices=small)
