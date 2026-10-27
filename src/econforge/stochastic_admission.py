"""Strict §20.2 A2 admission: V3 implementation plus separately bound full A1 wire."""
import json
from econforge.stochastic_contract import (
    need,TRUST_FIELDS,CONTENT_FIELDS,A1_SOURCE,A1_ACCEPTANCE_SHA256,
    ACCEPTANCE_SHA256,P38_SOURCE,P38_ORIGINAL,SPECIFICATIONS,canonical_json,digest)
from econforge.innovation_admission import require_fields,admit_trust,bound_digest
from econforge.innovation_wire import validate_wire as validate_A1
from econforge.qz_admission import same


def admit_a2(expected,content):
    need(expected is not None and content is not None,'TRUST_MISSING',
         'Separate trusted A2 enrollment and complete content required','missing')
    for value,fields,label in [(expected,TRUST_FIELDS,'A2 trust'),(content,CONTENT_FIELDS,'A2 content')]:
        need(type(value) is dict,'INVALID_INPUT',label+' must be a plain object')
        need(not (set(fields)-set(value)) and all(value[k] is not None for k in fields),
             'TRUST_MISSING','Missing required '+label+' content','missing')
        require_fields(value,fields,label)
    t,c=[json.loads(canonical_json(x)) for x in (expected,content)]
    need(t['schema']=='p39-stochastic-trust' and type(t['revision']) is int and t['revision']==1
         and t['stage']=='P3.9A2' and c['schema']=='p39-stochastic-content'
         and type(c['revision']) is int and c['revision']==1,
         'SOURCE_UNSUPPORTED','Only frozen A2 trust/content revision 1','unsupported')
    require_fields(t['enrollment'],{'authority','record_id','enrolled_at_utc','source_location'},'A2 enrollment')
    need(all(type(x) is str and bool(x.strip()) for x in t['enrollment'].values()),
         'TRUST_MISSING','Separately retained enrollment authority required','missing')
    require_fields(t['upstream'],{'stage','source_commit','result_sha256','acceptance_sha256'},'A1 V3 upstream')
    u=t['upstream']
    need(u['stage']=='P3.9A1' and u['source_commit']==A1_SOURCE
         and u['acceptance_sha256']==A1_ACCEPTANCE_SHA256,
         'ACCEPTANCE_MISSING','Only independently accepted exact A1 V3 source is admitted','missing')
    need(type(c['upstream_result']) is dict,'TRUST_MISSING','Complete accepted A1 result required','missing')
    # The entire wire is bound FIRST to a separately persisted expected identity.
    # Only after this gate can any nested provenance become retained trusted data.
    need(u['result_sha256'] is not None and u['result_sha256']!='','TRUST_MISSING','Separate accepted A1 expected digest missing','missing')
    bound_digest(c['upstream_result'],u['result_sha256'],'complete accepted A1 V3 wire')
    records=t['acceptance_records'];need(type(records) is list and bool(records),
        'ACCEPTANCE_MISSING','Complete accepted-source registry required','missing')
    require_fields(c['acceptance_records'],{ACCEPTANCE_SHA256,A1_ACCEPTANCE_SHA256},'A2 acceptance content registry')
    allowed={('P3.8',P38_ORIGINAL,ACCEPTANCE_SHA256),('P3.8',P38_SOURCE,ACCEPTANCE_SHA256),
             ('P3.9A1',A1_SOURCE,A1_ACCEPTANCE_SHA256)}
    actual=[]
    for row in records:
        require_fields(row,{'stage','source_commit','record_sha256','location'},'A2 acceptance entry')
        need(type(row['location']) is str and bool(row['location']),'INVALID_INPUT','Acceptance location required')
        key=(row['stage'],row['source_commit'],row['record_sha256']);actual.append(key)
        need(key in allowed,'ACCEPTANCE_MISSING','Unaccepted implementation or candidate acceptance','missing')
    need(len(actual)==len(allowed) and set(actual)==allowed,'ACCEPTANCE_MISSING',
         'Original P3.8, maintenance P3.8 and exact accepted V3 all required','missing')
    for h,text in c['acceptance_records'].items():
        need(type(text) is str and record_bytes_sha256(text)==h,'DIGEST_MISMATCH','Original acceptance record bytes differ')
    a1=c['upstream_result'];a1trust=a1.get('provenance')
    need(type(a1trust) is dict,'INVALID_INPUT','Accepted full A1 provenance required')
    require_fields(a1trust,TRUST_FIELDS,'accepted A1 provenance')
    shared=TRUST_FIELDS-{'stage','upstream','acceptance_records','enrollment'}
    need(all(same(t[k],a1trust[k]) for k in shared),'SOURCE_MISMATCH',
         'A2 trust and independently bound A1 source/context identities disagree')
    validate_A1(a1,expected=a1trust)
    need(a1['product_valid'],'LAW_MISSING','Successful joint accepted A1 product required','missing')
    need(a1['trust_sha256']==digest(a1trust),'DIGEST_MISMATCH','Retained A1 trust identity differs')
    # Reuse accepted A1/P3.8 admission without changing it or recovering new G/H/L.
    c1=json.loads(canonical_json(c));c1['upstream_result']=c1['p38_result']
    c1['acceptance_records']={ACCEPTANCE_SHA256:c1['acceptance_records'][ACCEPTANCE_SHA256]}
    t1,c1,specs=admit_trust(a1trust,c1)
    need(same(t['specifications'],SPECIFICATIONS),'SPEC_HASH_MISMATCH','A2 frozen specifications differ')
    return t,c,a1,t1,c1,specs


def record_bytes_sha256(text):
    # Raw record bytes, distinct from canonical JSON hashing. No dynamic import.
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
