"""Source-specific §20.6 Schur and §20.7 checked Lyapunov sufficient witness.

These internal numerical helpers issue diagnostics, never a complete-law claim.
No covariance estimation, moments product or response propagation is performed.
"""
import math
import warnings
from econforge.stochastic_contract import (
    InnovationError,need,finite,scalar,strict_margin,quality_bound,DELTA,FORCING_CHECKS)

LYAP_ARRAYS=('B','P0','P','Q','D0','D','P_eigenvalues','D_eigenvalues','P_vectors','D_vectors')
LYAP_SCALARS=('transform_sigma_min','transform_sigma_max','transform_margin','h_P','h_D',
 'P_correction','D_correction','P_norm2','Q_norm2','p_margin','d_margin','c_round',
 'cancellation_margin','equation_residual','P_eigen_residual','D_eigen_residual',
 'P_eigen_unitarity','D_eigen_unitarity','kappa_P','tau')


def wire(np,value):
    return np.stack((value.real,value.imag),axis=-1).tolist() if np.iscomplexobj(value) else value.tolist()


def fro(np,value):
    """Scaled Frobenius evaluation, preserving the frozen norm exactly in meaning."""
    need(bool(np.isfinite(value).all()),'NUMERICAL_NONFINITE','Nonfinite norm input','backend')
    if not value.size:return 0.0
    scale=float(np.max(np.abs(value)))
    if scale==0:return 0.0
    return scalar(np.linalg.norm(value/scale,ord='fro')*scale)


def backend(call):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        value=call()
    need(not observed,'BACKEND_WARNING','Forcing backend warning: '+
         '; '.join(str(w.message) for w in observed),'backend')
    return value


def classify_roots(roots):
    a,b=1.0-DELTA,1.0+DELTA
    classes=[];inner=[];outer=[];margin=[];distance=[]
    for root in roots:
        magnitude=scalar(abs(root));classes.append('STABLE' if magnitude<a else 'OUTSIDE' if magnitude>b else 'BOUNDARY')
        inner.append(scalar(abs(magnitude-a)));outer.append(scalar(abs(magnitude-b)))
        margin.append(scalar(a-magnitude));distance.append(min(inner[-1],outer[-1]))
    return classes,inner,outer,margin,distance


def spectral_norm(np,linalg,value):
    values=backend(lambda:linalg.svdvals(value,check_finite=True))
    finite(np,values,(len(value),))
    need(bool((values>=0).all()),'NUMERICAL_NONFINITE','Negative computed singular value','backend')
    return scalar(np.max(values)) if values.size else 0.0


def eigen_quality(np,linalg,M,tau):
    d,W=backend(lambda:linalg.eigh(M,lower=True,eigvals_only=False,driver='evd',check_finite=True))
    finite(np,d,(len(M),));finite(np,W,M.shape,complex_required=True)
    residual=scalar(fro(np,M-(W*d[None,:])@W.conj().T)/max(fro(np,M),1.0))
    unitarity=scalar(fro(np,W.conj().T@W-np.eye(len(M)))/max(len(M),1))
    norm=spectral_norm(np,linalg,M)
    return d,W,residual,unitarity,norm


def forcing(np,linalg,R_q,certificate_digest,gate,start,diagnostics):
    """Require every separate check; retain only finite, reached diagnostics."""
    m=len(R_q);finite(np,R_q,(m,m));need(m<=256,'INVALID_INPUT','Source dimension exceeds frozen bound')
    # Keep original source private even if an instrumented backend alters its input.
    R=R_q.copy();tau=1000*max(m,1)*2.0**-52
    schur=dict(R_q=R.tolist(),F=None,U=None,roots=[],classes=[],distance_inner=[],distance_outer=[],
        stable_margin=[],boundary_distance=[],reconstruction=None,unitarity=None,triangularity=None,
        tau=None if m==0 else tau,stable_count=0,outside_count=0,boundary_count=0,vacuous=m==0,
        source_certificate_sha256=certificate_digest)
    lyap={**dict.fromkeys(LYAP_ARRAYS),**dict.fromkeys(LYAP_SCALARS),
          'vacuous':m==0,'method':'bilinear'}
    diagnostics.extend([dict(id='A2.source_schur',kind='schur',values=schur),
                        dict(id='A2.source_lyapunov',kind='lyapunov',values=lyap)])
    if m==0:
        schur.update(F=[],U=[])
        for key in LYAP_ARRAYS:lyap[key]=[]
        for name in FORCING_CHECKS:gate(name,vacuous=True,diagnostic_id='A2.source_schur' if '.schur.' in name else 'A2.source_lyapunov')
        return schur,lyap
    start('A2.schur.quality')
    F,U=backend(lambda:linalg.schur(R.copy(),output='complex',sort=None,overwrite_a=False,check_finite=True))
    finite(np,F,(m,m),complex_required=True);finite(np,U,(m,m),complex_required=True)
    rec=scalar(fro(np,R-U@F@U.conj().T)/max(fro(np,R),1.0))
    uni=scalar(fro(np,U.conj().T@U-np.eye(m))/max(m,1))
    tri=scalar(fro(np,np.tril(F,-1))/max(fro(np,F),1.0))
    schur.update(F=wire(np,F),U=wire(np,U),reconstruction=rec,unitarity=uni,triangularity=tri)
    gate('A2.schur.quality',all(quality_bound(x,tau) for x in (rec,uni,tri)),
         diagnostic_id='A2.source_schur',reason='FORCING_QUALITY_FAILED')
    start('A2.schur.roots')
    roots=np.diag(F);classes,di,do,sm,bd=classify_roots(roots)
    schur.update(roots=wire(np,roots),classes=classes,distance_inner=di,distance_outer=do,
        stable_margin=sm,boundary_distance=bd,stable_count=classes.count('STABLE'),
        outside_count=classes.count('OUTSIDE'),boundary_count=classes.count('BOUNDARY'))
    gate('A2.schur.roots',schur['stable_count']==m,diagnostic_id='A2.source_schur',
         reason='FORCING_OUTSIDE' if schur['outside_count'] else 'FORCING_BOUNDARY',
         kind='outside' if schur['outside_count'] else 'uncertain')
    start('A2.lyapunov.transform')
    B=(R/(1.0-DELTA)).astype(np.complex128);finite(np,B,(m,m),complex_required=True)
    I=np.eye(m,dtype=np.complex128);lyap.update(B=wire(np,B),tau=tau)
    singular=backend(lambda:linalg.svdvals(B+I,check_finite=True));finite(np,singular,(m,))
    smax,smin=scalar(np.max(singular)),scalar(np.min(singular))
    need(0<=smin<=smax,'NUMERICAL_NONFINITE','Transform singular values invalid','backend')
    margin=scalar(smin/smax) if smax>0 else 0.0
    lyap.update(transform_sigma_min=smin,transform_sigma_max=smax,transform_margin=margin)
    gate('A2.lyapunov.transform',smax>0 and strict_margin(margin,tau),diagnostic_id='A2.source_lyapunov',reason='WITNESS_UNRESOLVED',kind='uncertain')
    start('A2.lyapunov.hermitian')
    P0=backend(lambda:linalg.solve_discrete_lyapunov(B.conj().T.copy(),I.copy(),method='bilinear'))
    finite(np,P0,(m,m),complex_required=True);lyap['P0']=wire(np,P0)
    hP=scalar(fro(np,P0-P0.conj().T)/max(fro(np,P0),1.0));lyap['h_P']=hP
    need(quality_bound(hP,tau),'WITNESS_UNRESOLVED','P0 Hermitian quality fails','uncertain')
    P=(P0+P0.conj().T)/2.0;finite(np,P,(m,m),complex_required=True)
    Q=B.conj().T@P@B;finite(np,Q,(m,m),complex_required=True)
    D0=P-Q;finite(np,D0,(m,m),complex_required=True)
    hD=scalar(fro(np,D0-D0.conj().T)/max(fro(np,D0),1.0));lyap.update(P=wire(np,P),Q=wire(np,Q),D0=wire(np,D0),h_D=hD,P_correction=fro(np,P-P0))
    need(quality_bound(hD,tau),'WITNESS_UNRESOLVED','D0 Hermitian quality fails','uncertain')
    D=(D0+D0.conj().T)/2.0;finite(np,D,(m,m),complex_required=True)
    lyap.update(D=wire(np,D),D_correction=fro(np,D-D0))
    gate('A2.lyapunov.hermitian',diagnostic_id='A2.source_lyapunov')
    start('A2.lyapunov.eigen_quality')
    pd,pv,pr,pu,pnorm=eigen_quality(np,linalg,P,tau)
    dd,dv,dr,du,dnorm=eigen_quality(np,linalg,D,tau)
    qnorm=spectral_norm(np,linalg,Q)
    lyap.update(P_eigenvalues=pd.tolist(),D_eigenvalues=dd.tolist(),P_vectors=wire(np,pv),D_vectors=wire(np,dv),
        P_eigen_residual=pr,D_eigen_residual=dr,P_eigen_unitarity=pu,D_eigen_unitarity=du,P_norm2=pnorm,Q_norm2=qnorm)
    gate('A2.lyapunov.eigen_quality',all(quality_bound(x,tau) for x in (pr,pu,dr,du)),diagnostic_id='A2.source_lyapunov',reason='WITNESS_UNRESOLVED',kind='uncertain')
    pmin,dmin=scalar(np.min(pd)),scalar(np.min(dd));pm=scalar(pmin/max(pnorm,1.0))
    lyap['p_margin']=pm
    gate('A2.lyapunov.positive_P',strict_margin(pm,tau),diagnostic_id='A2.source_lyapunov',reason='WITNESS_UNRESOLVED',kind='uncertain')
    kappa=scalar(pnorm/pmin);lyap['kappa_P']=kappa
    dm=scalar(dmin/max(scalar(pnorm+qnorm+1.0),1.0));lyap['d_margin']=dm
    gate('A2.lyapunov.positive_D',strict_margin(dm,tau),diagnostic_id='A2.source_lyapunov',reason='WITNESS_UNRESOLVED',kind='uncertain')
    start('A2.lyapunov.cancellation')
    try:
        C=np.abs(P)+np.abs(B).T@np.abs(P)@np.abs(B)
        cround=max(scalar(fro(np,C)+1.0),1.0)
    except (FloatingPointError,OverflowError,InnovationError):
        need(False,'WITNESS_UNRESOLVED','Absolute-product cancellation scale unrepresentable','uncertain')
    # Only C-screen unrepresentability has the explicit uncertainty exception.
    need(bool(np.isfinite(C).all()),'WITNESS_UNRESOLVED','Absolute-product cancellation scale nonfinite','uncertain')
    cm=scalar(dmin/cround);lyap.update(c_round=cround,cancellation_margin=cm)
    gate('A2.lyapunov.cancellation',strict_margin(cm,tau),diagnostic_id='A2.source_lyapunov',reason='WITNESS_UNRESOLVED',kind='uncertain')
    eq=scalar(fro(np,P-Q-I)/max(scalar(fro(np,P)+fro(np,Q)+math.sqrt(m)),1.0));lyap['equation_residual']=eq
    gate('A2.lyapunov.equation',quality_bound(eq,tau),diagnostic_id='A2.source_lyapunov',reason='WITNESS_UNRESOLVED',kind='uncertain')
    return schur,lyap
