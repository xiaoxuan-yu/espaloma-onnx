"""FP32 inference adapter for the supported Espaloma 0.3.2 checkpoint.

One prepared molecule per adapter. Graph preparation is outside forward.
Dense mean adjacency is intended for small ligand graphs, not large systems.
"""
import math
import torch


def mlp(net):
    assert all(isinstance(getattr(net, name), torch.nn.Module) or getattr(net, name) is torch.nn.functional.relu for name in net.exes)
    return torch.nn.Sequential(*[getattr(net, name) if isinstance(getattr(net, name), torch.nn.Module) else torch.nn.ReLU() for name in net.exes]).eval()


class TensorModel(torch.nn.Module):
    def __init__(self, model, graph):
        super().__init__()
        assert not model.training and graph.num_nodes("g") == 1
        assert model[2].__class__.__name__ == "JanossyPoolingWithSmirnoffImproper"
        self.input = model[0].f_in
        self.convs = torch.nn.ModuleList()
        for name in model[0]._sequential.exes:
            if name.startswith("d"):
                conv = getattr(model[0]._sequential, name).gn
                assert conv._aggre_type == "mean" and conv.norm is None and conv.activation is None
                self.convs.append(conv)
        n = graph.num_nodes("n1")
        src, dst = graph.edges(etype="n1_neighbors_n1")
        adjacency = torch.zeros(n, n)
        adjacency.index_put_((dst, src), torch.ones(src.shape), accumulate=True)
        adjacency /= adjacency.sum(1, keepdim=True).clamp(min=1)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("features", graph.nodes["n1"].data["h0"].clone())
        self.mlps = torch.nn.ModuleDict()
        self.heads = torch.nn.ModuleDict()
        self.levels = (2, 3, 4)
        for level in (1, 2, 3, 4):
            self.mlps[str(level)] = mlp(getattr(model[1], f"sequential_{level}"))
            for key in model[1].out_features[level]:
                self.heads[f"{level}_{key}"] = getattr(model[1], f"f_out_{level}_to_{key}")
        self.mlps["improper"] = mlp(model[2].sequential_n4_improper)
        self.heads["improper"] = model[2].f_out_n4_improper_to_k
        for level, size in (("2", 2), ("3", 3), ("4", 4), ("4_improper", 4)):
            count = graph.num_nodes("n" + level)
            indices = torch.empty(count, size, dtype=torch.long)
            for pos in range(size):
                s, d = graph.edges(etype=f"n1_as_{pos}_in_n{level}")
                assert len(s) == count and torch.equal(torch.sort(d).values, torch.arange(count))
                indices[d, pos] = s
            self.register_buffer("indices_" + level, indices)
        if "sum_q" in graph.nodes["g"].data:
            charge = graph.nodes["g"].data["sum_q"]
        elif "q_ref" in graph.nodes["n1"].data:
            charge = graph.nodes["n1"].data["q_ref"].sum(0, keepdim=True)
        else:
            charge = torch.zeros(1, 1)
        self.register_buffer("charge", charge.clone())
        self.eval()

    def forward(self, features, adjacency=None, indices_2=None, indices_3=None,
                indices_4=None, indices_4_improper=None, charge=None):
        adjacency = self.adjacency if adjacency is None else adjacency
        relations = (self.indices_2 if indices_2 is None else indices_2,
                     self.indices_3 if indices_3 is None else indices_3,
                     self.indices_4 if indices_4 is None else indices_4)
        improper = self.indices_4_improper if indices_4_improper is None else indices_4_improper
        charge = self.charge if charge is None else charge
        h = self.input(features)
        for conv in self.convs:
            h = torch.relu(conv.fc_self(h) + conv.fc_neigh(adjacency @ h))
        atom = self.mlps["1"](h)
        s = self.heads["1_s"](atom)
        e = self.heads["1_e"](atom)
        inv = s ** -1
        q = -e * inv + inv * ((charge + (e * inv).sum(0, keepdim=True)) / inv.sum(0, keepdim=True))
        outputs = {"n1_s": s, "n1_e": e, "n1_q": q}
        for level, indices in zip(self.levels, relations):
            x = h[indices]
            # Combine both permutations into one MLP call.
            z = self.mlps[str(level)](torch.cat((x.flatten(1), x.flip(1).flatten(1)), 0))
            front, back = z.chunk(2, 0)
            pooled = front + back
            if level == 4:
                outputs["n4_k"] = self.heads["4_k"](pooled)
            else:
                coefficients = self.heads[f"{level}_log_coefficients"](pooled)
                outputs[f"n{level}_log_coefficients"] = coefficients
                c = coefficients.exp()
                k = c[:, :1] + c[:, 1:]
                low, high = (1.5, 6.0) if level == 2 else (0.0, math.pi)
                outputs[f"n{level}_k"] = k
                outputs[f"n{level}_eq"] = (c[:, :1] * low + c[:, 1:] * high) / (k + 1e-7)
        x = h[improper]
        permutations = ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2))
        z = self.mlps["improper"](torch.cat([x[:, list(p)].flatten(1) for p in permutations], 0))
        pooled = torch.stack(z.chunk(3, 0), 0).sum(0)
        outputs["n4_improper_k"] = self.heads["improper"](pooled)
        return outputs
