import torch
import torch.nn.functional as F

# Values from the original computation chain where gelu produces -0.0
x = torch.tensor([-5.533906936645508, -5.531643390655518], dtype=torch.float32)

class BugModel(torch.nn.Module):
    def forward(self, x):
        return torch.sign(F.gelu(x))

model = BugModel()

with torch.no_grad():
    ref = model(x)  # eager

with torch.no_grad():
    aot_out = torch.compile(model, backend="aot_eager")(x)

with torch.no_grad():
    ind_out = torch.compile(model, mode="default")(x)

print(f"Input:  {x.tolist()}")
print(f"Eager:  {ref.tolist()}   (gelu produces -0.0, sign(-0.0)=0)")
print(f"aot:    {aot_out.tolist()}")
print(f"Induct: {ind_out.tolist()}   (gelu produces ~-1.65e-7, sign(tiny_neg)=-1)")
print(f"aot vs ref: same={torch._dynamo.utils.same(ref.double(), aot_out.double())}")
print(f"ind  vs ref: same={torch._dynamo.utils.same(ref.double(), ind_out.double())}")
print(f"max_diff (ind vs ref): {(ref.double() - ind_out.double()).abs().max().item():.6f}")

# Also the full chain from the original source
print("\n=== Full chain from original source ===")
torch.manual_seed(42)
v6 = torch.randn(23, 64, 62, dtype=torch.float32, device="cpu")

class FullChainModel(torch.nn.Module):
    def forward(self, x):
        a = torch.log2(x)
        b = torch.log2(a)
        c = torch.div(a, b)
        d = F.gelu(c)
        e = torch.sign(d)
        return torch.sigmoid(e)

full_model = FullChainModel()
with torch.no_grad():
    full_ref = full_model(v6)
full_ind = torch.compile(full_model, mode="default")
with torch.no_grad():
    full_ind_out = full_ind(v6)

print(f"Full chain ind vs ref: same={torch._dynamo.utils.same(full_ref.double(), full_ind_out.double())}")
print(f"max_diff: {(full_ref.double() - full_ind_out.double()).abs().max().item():.6f}")
