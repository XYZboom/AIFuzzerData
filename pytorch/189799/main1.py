import torch
import torch.nn as nn

class BugModule(nn.Module):
    def forward(self, x):
        v = torch.ceil(x)          # ceil(-0.5) = -0.0
        v = torch.add(v, torch.zeros_like(v))  # should be +0.0, inductor gives -0.0
        v = torch.reciprocal(v)    # reciprocal(+0.0) = +inf, reciprocal(-0.0) = -inf
        return v

x = torch.tensor([-0.5, -0.1, -0.9], dtype=torch.float32)

with torch.no_grad():
    eager_out = BugModule()(x)
with torch.no_grad():
    aot_out = torch.compile(BugModule(), backend="aot_eager")(x)
with torch.no_grad():
    ind_out = torch.compile(BugModule(), mode="default")(x)

print("Eager:  ", eager_out.tolist())   # [inf, inf, inf]
print("aot:    ", aot_out.tolist())     # [inf, inf, inf]
print("Induct: ", ind_out.tolist())     # [-inf, -inf, -inf]