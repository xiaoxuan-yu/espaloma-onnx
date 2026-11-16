"""Compare backend-neutral parameters with original deployment (reference-only OpenMM)."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import espaloma as esp
from openmm import unit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from espaloma_onnx import Predictor
predictor=Predictor.from_pretrained(ROOT/".artifacts/model")
model=torch.load(ROOT.parents[1]/".benchmark/espaloma-0.3.2.clean.pt",weights_only=False,map_location="cpu").eval()
torch.set_num_threads(8)
rows=[]
for smiles in ("CCO","CCCC","c1ccccc1","CC(=O)[O-]","CN1C=NC2=C1C(=O)N(C(=O)N2C)C"):
 with torch.no_grad():
  g=esp.Graph(smiles);model(g.heterograph)
  reference=esp.graphs.deploy.openmm_system_from_graph(g,forcefield="openff_unconstrained-2.1.1")
 result=predictor.create_forcefield(smiles);data=result.to_dict()
 for force in reference.getForces():
  name=type(force).__name__
  if name=="NonbondedForce":
   for i in range(force.getNumParticles()):
    q,s,e=force.getParticleParameters(i)
    np.testing.assert_allclose([data["atoms"]["charge_e"][i],result.sigma_nm[i],result.epsilon_kj_mol[i]],[q.value_in_unit(unit.elementary_charge),s.value_in_unit(unit.nanometer),e.value_in_unit(unit.kilojoule_per_mole)],rtol=1e-4,atol=1e-5)
   exceptions={tuple(x["indices"]):x for x in data["nonbonded"]["exceptions"]}
   assert len(exceptions)==force.getNumExceptions()
   for i in range(force.getNumExceptions()):
    a,b,qq,s,e=force.getExceptionParameters(i);entry=exceptions[tuple(sorted((a,b)))]
    expected_q=data["atoms"]["charge_e"][a]*data["atoms"]["charge_e"][b]*entry["coulomb_scale"]
    expected_e=(result.epsilon_kj_mol[a]*result.epsilon_kj_mol[b])**0.5*entry["lj_scale"]
    np.testing.assert_allclose([expected_q,expected_e],[qq.value_in_unit(unit.elementary_charge**2),e.value_in_unit(unit.kilojoule_per_mole)],rtol=1e-4,atol=1e-5)
  elif name in ("HarmonicBondForce","HarmonicAngleForce"):
   isbond=name=="HarmonicBondForce";term=data["bonds" if isbond else "angles"]
   lookup={tuple(idx):i for i,idx in enumerate(term["indices"])}
   count=force.getNumBonds() if isbond else force.getNumAngles()
   assert len(lookup)==count
   for i in range(count):
    values=force.getBondParameters(i) if isbond else force.getAngleParameters(i)
    atoms=tuple(values[:-2]);atoms=min(atoms,atoms[::-1]);j=lookup[atoms]
    eq,k=values[-2:];eq_unit=unit.nanometer if isbond else unit.radian
    np.testing.assert_allclose([term["equilibrium"][j],term["k"][j]],[eq.value_in_unit(eq_unit),k.value_in_unit(unit.kilojoule_per_mole/eq_unit**2)],rtol=1e-4,atol=1e-5)
  elif name=="PeriodicTorsionForce":
   terms=data["proper_torsions"]+data["improper_torsions"]
   assert len(terms)==force.getNumTorsions()
   for i,term in enumerate(terms):
    a,b,c,d,n,phase,k=force.getTorsionParameters(i)
    assert term["indices"]==[a,b,c,d] and term["periodicity"]==n
    np.testing.assert_allclose([term["phase_rad"],term["k_kj_mol"]],[phase.value_in_unit(unit.radian),k.value_in_unit(unit.kilojoule_per_mole)],rtol=1e-4,atol=1e-5)
 row=dict(smiles=smiles,atoms=len(result.sigma_nm),status="pass",checks="charges,LJ,bonds,angles,proper/improper terms,nonbonded scaling exceptions")
 rows.append(row);print(json.dumps(row),flush=True)
(ROOT/"benchmarks/results/full-forcefield.json").write_text(json.dumps(rows,indent=2))
