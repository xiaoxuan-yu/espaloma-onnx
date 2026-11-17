"""Optional OpenMM backend for the backend-neutral parameter schema."""
import math


def create_system(parameters, *, nonbonded_method="NoCutoff", cutoff_nm=None,
                  box_vectors_nm=None, remove_center_of_mass=True):
    try:
        import openmm
    except ImportError as exc:
        raise ImportError("Install espaloma-onnx[openmm] to construct an OpenMM System") from exc
    if parameters.get("schema_version") != 1:
        raise ValueError("Unsupported force-field parameter schema")
    expected_units = {"charge":"elementary_charge", "length":"nm", "energy":"kJ/mol", "angle":"radian", "bond_k":"kJ/mol/nm^2", "angle_k":"kJ/mol/radian^2"}
    if parameters.get("units") != expected_units:
        raise ValueError("Unsupported parameter units")
    atoms = parameters["atoms"]
    n = len(atoms["atomic_numbers"])
    if any(len(atoms[key]) != n for key in ("charge_e", "sigma_nm", "epsilon_kj_mol")):
        raise ValueError("Inconsistent per-atom parameter lengths")
    rules = parameters["nonbonded"]
    if rules["vdw"].get("combining_rules") != "Lorentz-Berthelot":
        raise ValueError("OpenMM backend requires Lorentz-Berthelot LJ combining rules")
    allowed = {"NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "Ewald"}
    if nonbonded_method not in allowed:
        raise ValueError("Unsupported nonbonded_method")
    periodic = nonbonded_method in {"CutoffPeriodic", "PME", "Ewald"}
    if periodic and box_vectors_nm is None:
        raise ValueError("Periodic methods require box_vectors_nm")
    system = openmm.System()
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    masses = atoms.get("mass_da", [table.GetAtomicWeight(int(z)) for z in atoms["atomic_numbers"]])
    for mass in masses:
        system.addParticle(float(mass))
    if box_vectors_nm is not None:
        if len(box_vectors_nm) != 3 or any(len(row) != 3 for row in box_vectors_nm):
            raise ValueError("box_vectors_nm must be three vectors of length three")
        system.setDefaultPeriodicBoxVectors(*(openmm.Vec3(*row) for row in box_vectors_nm))
    bonds = openmm.HarmonicBondForce()
    for idx,k,eq in zip(parameters["bonds"]["indices"],parameters["bonds"]["k"],parameters["bonds"]["equilibrium"]):
        bonds.addBond(*idx,eq,k)
    system.addForce(bonds)
    angles = openmm.HarmonicAngleForce()
    for idx,k,eq in zip(parameters["angles"]["indices"],parameters["angles"]["k"],parameters["angles"]["equilibrium"]):
        angles.addAngle(*idx,eq,k)
    system.addForce(angles)
    torsions = openmm.PeriodicTorsionForce()
    for term in parameters["proper_torsions"] + parameters["improper_torsions"]:
        torsions.addTorsion(*term["indices"],term["periodicity"],term["phase_rad"],term["k_kj_mol"])
    system.addForce(torsions)
    for constraint in parameters["constraints"]:
        system.addConstraint(*constraint["indices"],constraint["distance_nm"])
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(getattr(openmm.NonbondedForce,nonbonded_method))
    if nonbonded_method != "NoCutoff":
        cutoff = cutoff_nm if cutoff_nm is not None else rules["vdw"].get("cutoff_nm")
        if cutoff is None or cutoff <= 0:
            raise ValueError("A positive cutoff_nm is required")
        nonbonded.setCutoffDistance(float(cutoff))
        width = rules["vdw"].get("switch_width_nm",0.)
        if width:
            if width >= cutoff:
                raise ValueError("LJ switching width must be smaller than the cutoff")
            nonbonded.setUseSwitchingFunction(True)
            nonbonded.setSwitchingDistance(cutoff-width)
    nonbonded.setUseDispersionCorrection(periodic)
    for q,s,e in zip(atoms["charge_e"],atoms["sigma_nm"],atoms["epsilon_kj_mol"]):
        nonbonded.addParticle(q,s,e)
    for term in rules["exceptions"]:
        i,j=term["indices"]
        q=atoms["charge_e"][i]*atoms["charge_e"][j]*term["coulomb_scale"]
        sigma=(atoms["sigma_nm"][i]+atoms["sigma_nm"][j])*0.5
        epsilon=math.sqrt(atoms["epsilon_kj_mol"][i]*atoms["epsilon_kj_mol"][j])*term["lj_scale"]
        nonbonded.addException(i,j,q,sigma,epsilon)
    system.addForce(nonbonded)
    if remove_center_of_mass:
        system.addForce(openmm.CMMotionRemover())
    return system
