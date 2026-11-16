import argparse
import json
from .predictor import Predictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", choices=["huggingface", "modelscope"])
    parser.add_argument("--revision")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--full-forcefield", action="store_true")
    parser.add_argument("--forcefield", default="openff_unconstrained-2.1.1.offxml")
    args = parser.parse_args()
    predictor = Predictor.from_pretrained(args.model, source=args.source, revision=args.revision, threads=args.threads)
    if args.full_forcefield:
        print(json.dumps(predictor.create_forcefield(args.smiles, forcefield=args.forcefield).to_dict()))
        return
    result = predictor(args.smiles)
    print(json.dumps({"parameters": {k:v.tolist() for k,v in result.parameters.items()},
                      "indices": {k:v.tolist() for k,v in result.indices.items()}, "total_charge":result.total_charge}))
