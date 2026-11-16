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
    args = parser.parse_args()
    result = Predictor.from_pretrained(args.model, source=args.source, revision=args.revision, threads=args.threads)(args.smiles)
    print(json.dumps({"parameters": {k:v.tolist() for k,v in result.parameters.items()},
                      "indices": {k:v.tolist() for k,v in result.indices.items()}, "total_charge":result.total_charge}))
