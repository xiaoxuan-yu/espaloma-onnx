"""RDKit/NumPy input preparation and ORT inference for Espaloma 0.3.2.

Atom numbering is RDKit parse order with explicit hydrogens appended. Mirrors
OpenFF RDKit MDL aromaticity and isotope removal in its to_rdkit round-trip.
"""
import numpy as np
from rdkit import Chem


def prepare(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Invalid or empty SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    flags = Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS ^ Chem.SANITIZE_SETAROMATICITY
    Chem.SanitizeMol(mol, flags)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.AssignStereochemistry(mol)
    mol = Chem.AddHs(mol)
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("Radicals are not supported")
    # Recreate atoms/bonds as OpenFF to_rdkit does (isotope is not retained).
    rebuilt = Chem.RWMol()
    for old in mol.GetAtoms():
        atom = Chem.Atom(old.GetAtomicNum())
        atom.SetFormalCharge(old.GetFormalCharge())
        atom.SetIsAromatic(old.GetIsAromatic())
        atom.SetNoImplicit(True)
        rebuilt.AddAtom(atom)
    for bond in mol.GetBonds():
        rebuilt.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    mol = rebuilt.GetMol()
    Chem.SanitizeMol(mol, flags)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    n = mol.GetNumAtoms()
    features = np.zeros((n, 114), dtype=np.float32)
    hybridizations = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    for atom in mol.GetAtoms():
        i, z = atom.GetIdx(), atom.GetAtomicNum()
        if not 0 < z < 100:
            raise ValueError("Atomic number outside feature vocabulary")
        features[i, z] = 1
        features[i, 100:109] = [atom.GetTotalDegree(), atom.GetIsAromatic(), atom.GetMass()] + [atom.IsInRingSize(k) for k in range(3, 9)]
        hyb = atom.GetHybridization()
        if hyb in hybridizations:
            features[i, 109 + hybridizations.index(hyb)] = 1
        elif hyb != Chem.HybridizationType.S:
            raise ValueError("Unsupported hybridization: " + str(hyb))
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    neighbors = [[a.GetIdx() for a in atom.GetNeighbors()] for atom in mol.GetAtoms()]
    angles, propers = set(), set()
    for i in range(n):
        for j in neighbors[i]:
            for k in neighbors[j]:
                if k == i:
                    continue
                angles.add((i,j,k) if i < k else (k,j,i))
                for l in neighbors[k]:
                    if l in (i,j):
                        continue
                    propers.add((i,j,k,l) if i < l else (l,k,j,i))
    query = Chem.MolFromSmarts("[*:1]~[X3:2](~[*:3])~[*:4]")
    impropers = mol.GetSubstructMatches(query, uniquify=False, maxMatches=1000000)
    def array(rows, width, duplicate=False):
        result = np.asarray(rows, dtype=np.int64).reshape(-1, width)
        return np.concatenate((result, result[:, ::-1]), 0) if duplicate else result
    adjacency = np.zeros((n,n), dtype=np.float32)
    for i,j in bonds:
        adjacency[i,j] = adjacency[j,i] = 1
    adjacency /= np.maximum(adjacency.sum(1, keepdims=True), 1)
    return dict(features=features, adjacency=adjacency,
                indices_2=array(bonds,2,True), indices_3=array(sorted(angles),3,True),
                indices_4=array(sorted(propers),4,True), indices_4_improper=array(impropers,4),
                charge=np.array([[sum(a.GetFormalCharge() for a in mol.GetAtoms())]], dtype=np.float32))


class Predictor:
    def __init__(self, path, threads=8):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, smiles):
        return dict(zip(self.output_names, self.session.run(None, prepare(smiles))))


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles")
    parser.add_argument("--model", default=".benchmark/espaloma-dynamic.onnx")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    prediction = Predictor(args.model, args.threads)(args.smiles)
    print(json.dumps({key: value.tolist() for key, value in prediction.items()}))
