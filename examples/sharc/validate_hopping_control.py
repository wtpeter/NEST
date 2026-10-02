"""CPU synthetic-interface checks for initialized SHARC hopping controls.

This tests the SHARC engine and restart guard, not molecular electronic structure.
"""
import argparse
import json
import shlex
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sharc',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
binary=args.sharc.resolve()
report=dict(status='running',cases=[],scope='Synthetic prescribed overlaps and zero forces; engine control regression only')
def save():
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    for ns in (3,5,6):
        folder=OUT/f'{ns}S3T';folder.mkdir();(folder/'QM').mkdir();n=ns+9
        for name in ('geom','veloc'):shutil.copy2(BASE/name,folder/name)
        input_text=(BASE/'input').read_text().replace('nstates 3 0 3',f'nstates {ns} 0 3')
        input_text=input_text.replace('state 2 mch','state 3 mch').replace('ezero -38.570985220764186','ezero 0')
        (folder/'input').write_text(input_text)
        fixture=folder/'QM/fixture.py'
        fixture.write_text(f'''from pathlib import Path
import numpy as np
from nest.interfaces.sharc_nttda import write_qmout
n={n};ns={ns};nat=3
energies=list(.01*np.arange(ns))+[.008,.016,.024]*3
ov=np.eye(n);angle=.15
ov[1:3,1:3]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
result=dict(h=np.diag(energies).astype(complex),grad=np.zeros((n,nat,3)),
    nac=np.zeros((n,n,nat,3)),dm=np.zeros((3,n,n),complex),overlap=ov,phases=np.ones(n))
write_qmout(Path('QM.out'),result,dict(dm=[],grad=[],overlap=[],phases=[]),.01)
''')
        run=folder/'QM/runQM.sh'
        run.write_text(f'#!/bin/sh\ncd QM\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fixture))}\n');run.chmod(run.stat().st_mode|0o111)
        with (folder/'run.log').open('w') as log:
            subprocess.run([str(binary),'input'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
        lines=(folder/'restart.ctrl').read_text().splitlines()
        index=next(i for i,x in enumerate(lines) if '! hopping_procedure' in x)
        army=int(lines[index+5].split()[0]);assert army==0
        data=(folder/'output.dat').read_text().splitlines();randoms=[];probs=[];step=-1
        for i,line in enumerate(data):
            if line.startswith('! 0 Step'):step=int(data[i+1])
            if step>0 and line.startswith('! 9 Random'):randoms.append(float(data[i+1]))
            if step>0 and line.startswith('! 6 Hopping'):
                probs.append(sum(float(x) for x in data[i+1:i+1+n]))
        assert randoms and all(x>0 for x in randoms)
        assert max(probs)>1e-6
        report['cases'].append(dict(ns=ns,nt=3,army_ants=army,random_draws=randoms,probability_sums=probs));save()
    valid=OUT/'valid-restart';shutil.copytree(OUT/'5S3T',valid)
    (valid/'input').write_text((valid/'input').read_text().replace('tmax 0.2','tmax 0.3')+'\nrestart\n')
    with (valid/'run.log').open('w') as log:
        subprocess.run([str(binary),'input'],cwd=valid,stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
    if 'DM' not in (valid/'QM/QM.in').read_text().splitlines():
        raise RuntimeError('Restart did not restore the default dipole request')
    report['valid_restart_requested_dipoles']=True;save()
    bad=OUT/'invalid-restart';shutil.copytree(OUT/'5S3T',bad)
    lines=(bad/'restart.ctrl').read_text().splitlines();index=next(i for i,x in enumerate(lines) if '! hopping_procedure' in x)
    lines[index+5]='1986356319 ! army_ants';(bad/'restart.ctrl').write_text('\n'.join(lines)+'\n')
    (bad/'input').write_text((bad/'input').read_text().replace('tmax 0.2','tmax 0.3')+'\nrestart\n')
    result=subprocess.run([str(binary),'input'],cwd=bad,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'Invalid army_ants control in restart' in result.stderr+result.stdout
    report.update(status='completed',invalid_restart_rejected=True);save()
except Exception as error:
    report.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise
print(json.dumps(report),flush=True)
