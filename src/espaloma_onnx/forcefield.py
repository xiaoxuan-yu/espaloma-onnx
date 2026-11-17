"""Backend-neutral complete parameter assignment using OpenFF LJ typing."""
from dataclasses import dataclass
import math
import numpy as np

HARTREE_KJ_MOL = 2625.4996394798254
BOHR_NM = 0.0529177210903

@dataclass
class ForceFieldResult:
    prediction: object
    data: dict

    @property
    def sigma_nm(self):
        return np.asarray(self.data["atoms"]["sigma_nm"])

    @property
    def epsilon_kj_mol(self):
        return np.asarray(self.data["atoms"]["epsilon_kj_mol"])

    def to_openmm_system(self, **kwargs):
        from .openmm_backend import create_system
        return create_system(self.data, **kwargs)

    def to_dict(self):
        return self.data

    def write_json(self, path):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.data, indent=2)+"\n")


def create_forcefield(predictor, smiles, *, forcefield="openff_unconstrained-2.1.1.offxml"):
    try:
        from openff.units import unit
        from openff.toolkit import Molecule, ForceField
        from openff.toolkit.utils import RDKitToolkitWrapper, ToolkitRegistry
    except ImportError as exc:
        raise ImportError("Complete parameters require OpenFF Toolkit base and force-field data; see docs/full-forcefield.md") from exc
    registry = ToolkitRegistry([RDKitToolkitWrapper()])
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True, toolkit_registry=registry)
    from rdkit import Chem
    if len(Chem.GetMolFrags(molecule.to_rdkit())) != 1:
        raise ValueError("Complete parameter assignment expects one connected molecule")
    prediction = predictor(smiles)
    if molecule.n_atoms != len(prediction.atom_features):
        raise ValueError("OpenFF and inference atom counts differ")
    bonds = {(b.atom1_index,b.atom2_index) for b in molecule.bonds}
    expected = {tuple(map(int,row)) for row in prediction.indices["indices_2"]}
    if bonds | {(j,i) for i,j in bonds} != expected:
        raise ValueError("OpenFF and inference atom/bond ordering differ")
    ff = ForceField(forcefield)
    handlers = set(ff.registered_parameter_handlers)
    if any(name in handlers and len(ff.get_parameter_handler(name).parameters) for name in ("VirtualSites","GBSA")):
        raise ValueError("Virtual sites and implicit solvent are not supported")
    all_labels = ff.label_molecules(molecule.to_topology())[0]
    labels = all_labels["vdW"]
    sigma, epsilon, ids = [], [], []
    for i in range(molecule.n_atoms):
        parameter = labels[(i,)]
        s = parameter.sigma
        if s is None:
            s = 2 * parameter.rmin_half / 2**(1/6)
        sigma.append(float(s.m_as(unit.nanometer)))
        epsilon.append(float(parameter.epsilon.m_as(unit.kilojoule/unit.mole)))
        ids.append(parameter.id)
    def rules(handler):
        result = {"scale"+str(n): float(getattr(handler,"scale"+str(n))) for n in (12,13,14,15)}
        for name in ("combining_rules","potential","method","periodic_method","nonperiodic_method","exception_potential"):
            value = getattr(handler,name,None)
            if value is not None:result[name] = str(value)
        for name in ("cutoff","switch_width"):
            value = getattr(handler,name,None)
            if value is not None:result[name+"_nm"] = float(value.m_as(unit.nanometer))
        return result
    vdw = rules(ff.get_parameter_handler("vdW"))
    electrostatics = rules(ff.get_parameter_handler("Electrostatics"))
    params = prediction.parameters
    # Canonical rows reproduce upstream deployment's harmonic and torsion factors.
    terms = {}
    for level in (2,3):
        idx = prediction.indices["indices_"+str(level)]
        keep = idx[:,0] < idx[:,-1]
        factor = HARTREE_KJ_MOL/(BOHR_NM**2 if level==2 else 1.)
        terms["bonds" if level==2 else "angles"] = dict(indices=idx[keep].tolist(),
            k=(params[f"n{level}_k"][keep,0]*factor).tolist(),
            equilibrium=(params[f"n{level}_eq"][keep,0]*(BOHR_NM if level==2 else 1.)).tolist())
    for key,name,factor in (("4","proper_torsions",1.),("4_improper","improper_torsions",0.5)):
        output=[]
        for indices,ks in zip(prediction.indices["indices_"+key],params["n"+key+"_k"]):
            atoms=list(map(int,indices))
            if key=="4" and atoms[0]>=atoms[-1]:continue
            for periodicity,k in enumerate(ks,1):
                k=float(k)
                if k:output.append(dict(indices=atoms,periodicity=periodicity,phase_rad=math.pi if k<0 else 0.,k_kj_mol=abs(k)*factor*HARTREE_KJ_MOL))
        terms[name]=output
    # Graph-distance exceptions: 1-2/1-3 exclusions and 1-4 scaling (shortest path).
    adjacency=[set() for _ in range(molecule.n_atoms)]
    for i,j in bonds:adjacency[i].add(j);adjacency[j].add(i)
    exceptions=[]
    for i in range(molecule.n_atoms):
        seen={i};front={i}
        for distance in range(1,5):
            front={j for v in front for j in adjacency[v]}-seen
            seen.update(front)
            for j in sorted(front):
                if j<=i:continue
                label="scale"+str(11+distance)
                # distance 1->scale12, 2->scale13, 3->scale14, 4->scale15
                lj,coulomb=vdw[label],electrostatics[label]
                if lj!=1. or coulomb!=1.:exceptions.append(dict(indices=[i,j],lj_scale=lj,coulomb_scale=coulomb))
    constraints=[]
    bond_lengths={tuple(sorted(idx)):eq for idx,eq in zip(terms["bonds"]["indices"],terms["bonds"]["equilibrium"])}
    for indices,parameter in all_labels.get("Constraints",{}).items():
        distance=parameter.distance
        length=float(distance.m_as(unit.nanometer)) if distance is not None else bond_lengths[tuple(sorted(indices))]
        constraints.append(dict(indices=list(indices),distance_nm=length))
    data=dict(schema_version=1,forcefield=forcefield,smiles=smiles,
        units=dict(charge="elementary_charge",length="nm",energy="kJ/mol",angle="radian",bond_k="kJ/mol/nm^2",angle_k="kJ/mol/radian^2"),
        functional_forms=dict(bonds="0.5*k*(r-equilibrium)^2",angles="0.5*k*(theta-equilibrium)^2",torsions="k*(1+cos(periodicity*phi-phase))",lj="4*epsilon*((sigma/r)^12-(sigma/r)^6)"),
        atoms=dict(mass_da=[float(a.mass.m_as(unit.dalton)) for a in molecule.atoms],atomic_numbers=[a.atomic_number for a in molecule.atoms],charge_e=prediction.parameters["n1_q"].flatten().tolist(),sigma_nm=sigma,epsilon_kj_mol=epsilon,vdw_parameter_ids=ids),
        **terms,constraints=constraints,nonbonded=dict(vdw=vdw,electrostatics=electrostatics,exceptions=exceptions,default_lj_scale=1.,default_coulomb_scale=1.))
    return ForceFieldResult(prediction,data)
