import torch
import torch.nn as nn
import torch.nn.functional as F

class SqrtNegativeNaN(nn.Module):
    def forward(self, x):
        return torch.sqrt(x)

def check_nan_sign_bit(tensor, label):
    int_view = tensor.view(torch.int32)
    nan_mask = torch.isnan(tensor)
    if nan_mask.any():
        nan_bits = int_view[nan_mask]
        sign_bit = (nan_bits >> 31) & 1
        print(f"  {label}: {nan_mask.sum().item()} NaNs, "
              f"sign_bit={sign_bit.unique().tolist()}, "
              f"first_nan=0x{nan_bits[0].item():08x}")
    else:
        print(f"  {label}: no NaNs")

# Test 1: Single negative value
x = torch.tensor([-1.0], dtype=torch.float32, device="cpu")
model = SqrtNegativeNaN()

with torch.no_grad():
    eager = model(x)
with torch.no_grad():
    aot = torch.compile(model, backend="aot_eager")(x)
with torch.no_grad():
    ind = torch.compile(model, mode="default")(x)

check_nan_sign_bit(eager, "eager")
check_nan_sign_bit(aot, "aot_eager")
check_nan_sign_bit(ind, "inductor")

# Test 2: Mixed tensor
torch.manual_seed(42)
x2 = torch.randn(24, 23, 13, dtype=torch.float32, device="cpu")

with torch.no_grad():
    eager2 = model(x2)
with torch.no_grad():
    aot2 = torch.compile(model, backend="aot_eager")(x2)
with torch.no_grad():
    ind2 = torch.compile(model, mode="default")(x2)

print("\n--- Mixed tensor ---")
check_nan_sign_bit(eager2, "eager")
check_nan_sign_bit(aot2, "aot_eager")
check_nan_sign_bit(ind2, "inductor")

# Positive sqrt precision difference
pos_mask = x2 > 0
pos_diff = (eager2[pos_mask] - ind2[pos_mask]).abs()
print(f"Positive sqrt max diff: {pos_diff.max().item():.2e}")
