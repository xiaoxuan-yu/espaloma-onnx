"""Optional System backend versus original Espaloma deployment."""
import json
from pathlib import Path
import sys
import numpy as np
import openmm
from openmm import unit
from rdkit.Chem import AllChem
import torch
import espaloma as esp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from espaloma_onnx import Predictor
predict=Predictor.from_pretrained(ROOT/".artifacts/model")
net=torch.load(ROOT.parents[1]/".benchmark/espaloma-0.3.2.clean.pt",map_location="cpu",weights_only=False).eval()
torch.set_num_threads(8)
rows=[]
for smiles in ("CCO","CCCC","c1ccccc1","CC(=O)[O-]","CN1C=NC2=C1C(=O)N(C(=O)N2C)C"):
 with torch.no_grad():
  graph=esp.Graph(smiles);net(graph.heterograph)
  reference=esp.graphs.deploy.openmm_system_from_graph(graph,forcefield="openff_unconstrained-2.1.1")
 ff=predict.create_forcefield(smiles)
 system=ff.to_openmm_system()
 assert system.getNumParticles()==reference.getNumParticles()
 for i in range(system.getNumParticles()):
  np.testing.assert_allclose(system.getParticleMass(i).value_in_unit(unit.dalton),reference.getParticleMass(i).value_in_unit(unit.dalton),atol=1e-6)
 nb=next(f for f in system.getForces() if isinstance(f,openmm.NonbondedForce))
 refnb=next(f for f in reference.getForces() if isinstance(f,openmm.NonbondedForce))
 assert nb.getNumExceptions()==refnb.getNumExceptions()
 expected={tuple(sorted(refnb.getExceptionParameters(i)[:2])):refnb.getExceptionParameters(i)[2:] for i in range(refnb.getNumExceptions())}
 for i in range(nb.getNumExceptions()):
  a,b,q,s,e=nb.getExceptionParameters(i);rq,rs,re=expected[tuple(sorted((a,b)))]
  # Sigma is physically immaterial when epsilon=0 (upstream uses 1 nm).
  np.testing.assert_allclose(q.value_in_unit(rq.unit),rq._value,rtol=1e-4,atol=1e-5)
  np.testing.assert_allclose(e.value_in_unit(re.unit),re._value,rtol=1e-4,atol=1e-5)
  if re._value:np.testing.assert_allclose(s.value_in_unit(rs.unit),rs._value,rtol=1e-5)
 mol=graph.mol.to_rdkit();AllChem.EmbedMolecule(mol,randomSeed=42)
 positions=np.asarray(mol.GetConformer().GetPositions())*0.1
 energies=[];forces=[]
 # Verify portable XML survives round-trip.
 system=openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
 for candidate in (reference,system):
  integrator=openmm.VerletIntegrator(0.001)
  context=openmm.Context(candidate,integrator,openmm.Platform.getPlatformByName("Reference"))
  context.setPositions(positions);state=context.getState(getEnergy=True,getForces=True)
  energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
  forces.append(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
  del context,integrator
 np.testing.assert_allclose(energies[0],energies[1],rtol=1e-4,atol=1e-3)
 np.testing.assert_allclose(forces[0],forces[1],rtol=1e-4,atol=1e-3)
 row=dict(smiles=smiles,status="pass",reference_energy_kj_mol=energies[0],backend_energy_kj_mol=energies[1],max_force_error_kj_mol_nm=float(np.abs(forces[0]-forces[1]).max()))
 print(json.dumps(row),flush=True);rows.append(row)
(ROOT/"benchmarks/results/openmm-backend.json").write_text(json.dumps(rows,indent=2))
