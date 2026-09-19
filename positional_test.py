"""Check that DiffMeanPositional keeps column information from training to inference.

Runs the real DiffMeanPositional.train and PositionwisePromptAdditionIntervention.forward
on a tiny hand-built batch instead of Gemma, so every expected value can be worked out by
hand. CPU only, no model weights.

    .venv/bin/python positional_test.py      (or: pytest positional_test.py)

The fake "model"
----------------
Token ids: 0 = pad, 1 = token in a positive example, 2 = token in a negative example.
The stubbed layer output at column c has 3 dims:

    positive token -> [1, c, 0]     dim 1 records which column the token sat in
    negative token -> [0, 0, 1]
    pad            -> [0, 0, 0]

So a column's difference of means is [f_pos, f_pos * c, -f_neg], where f_pos / f_neg is
the fraction of positive / negative rows that have a real token in that column (padding
is deliberately not masked out). Any vector read back has dim1 / dim0 == the column it
came from, so a vector that ends up in the wrong place is obvious.
"""
from types import SimpleNamespace

import torch

import axbench.models.mean as mean_mod
from axbench.models.mean import DiffMeanPositional, LogisticRegressionModel
from axbench.models.interventions import PositionwisePromptAdditionIntervention

PAD, POS, NEG = 0, 1, 2
H = 3


def fake_layer_output(model, layer, inputs):
    ids = inputs["input_ids"].long()
    col = torch.arange(ids.shape[1]).expand_as(ids).float()
    is_pos, is_neg = (ids == POS).float(), (ids == NEG).float()
    return torch.stack([is_pos, is_pos * col, is_neg], dim=-1)


def left_padded_batch(rows, width):
    """rows: list of (label, real_length). Left-pads each row to `width`."""
    ids, mask = [], []
    for label, length in rows:
        tok = POS if label == 1 else NEG
        ids.append([PAD] * (width - length) + [tok] * length)
        mask.append([0] * (width - length) + [1] * length)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.int),
        "attention_mask": torch.tensor(mask, dtype=torch.int),
        "labels": torch.tensor([label for label, _ in rows], dtype=torch.int),
    }


def train_positional(batches, num_positions):
    """Run the real DiffMeanPositional.train with the model swapped for the stub above."""
    mean_mod.gather_residual_activations = fake_layer_output
    m = object.__new__(DiffMeanPositional)
    m.training_args = SimpleNamespace(n_epochs=1, num_positions=num_positions)
    m.device = "cpu"
    m.model = SimpleNamespace(config=SimpleNamespace(hidden_size=H))
    m.layer = 0
    m.ax = LogisticRegressionModel(H, 1)
    m.make_dataloader = lambda examples, **kw: batches
    DiffMeanPositional.train(m, None)
    return m.positional_weight  # [num_positions, H], column order


def apply_positional(stack, prompt_width, factor=2.0, max_act=3.0, batch_size=2):
    """Run the real intervention on an all-zero prefill; the output is exactly the delta."""
    iv = PositionwisePromptAdditionIntervention(
        embed_dim=H, low_rank_dimension=1, num_positions=stack.shape[0])
    iv.proj_weight.data = stack.unsqueeze(0)  # one concept
    sub = {"idx": torch.zeros(batch_size, dtype=torch.long),
           "mag": torch.full((batch_size,), factor),
           "max_act": torch.full((batch_size,), max_act)}
    return iv(torch.zeros(batch_size, prompt_width, H), subspaces=sub)


def unit(v):
    return v / (v.norm(dim=-1, keepdim=True) + torch.finfo(v.dtype).eps)


# Training data: 3 positives and 3 negatives of different lengths, width W = 6, split over
# two batches (the loop has to accumulate across batches). Real tokens per column:
#
#   column:        0  1  2  3  4  5
#   pos len 6:     P  P  P  P  P  P
#   pos len 4:     .  .  P  P  P  P
#   pos len 2:     .  .  .  .  P  P
#   neg len 6:     N  N  N  N  N  N
#   neg len 3:     .  .  .  N  N  N
#   neg len 1:     .  .  .  .  .  N
#
#   f_pos:        1/3 1/3 2/3 2/3  1   1
#   f_neg:        1/3 1/3 1/3 2/3 2/3  1
W = 6
TRAIN = [
    left_padded_batch([(1, 6), (1, 4), (0, 3)], W),
    left_padded_batch([(1, 2), (0, 6), (0, 1)], W),
]

# diff of means per column = [f_pos, f_pos * c, -f_neg], worked out from the table above
EXPECTED_DIFF = torch.tensor([
    [1/3, 0 * 1/3, -1/3],  # column 0
    [1/3, 1 * 1/3, -1/3],  # column 1
    [2/3, 2 * 2/3, -1/3],  # column 2
    [2/3, 3 * 2/3, -2/3],  # column 3
    [1,   4 * 1,   -2/3],  # column 4
    [1,   5 * 1,   -1  ],  # column 5 (last token)
])


def test_training_stores_columns_in_order():
    # num_positions = 4 < W: the stack is exactly the last 4 columns, left to right.
    P = 4
    stack = train_positional(TRAIN, P)
    assert stack.shape == (P, H)
    assert torch.allclose(stack, unit(EXPECTED_DIFF[W - P:]), atol=1e-6)
    # dim1 / dim0 recovers the source column: slot j <- column W-P+j
    assert torch.allclose(stack[:, 1] / stack[:, 0], torch.tensor([2., 3., 4., 5.]))


def test_training_narrow_batch_pads_leading_slots():
    # num_positions = 8 > W = 6: columns 0..5 fill the last 6 slots, the first 2 stay zero.
    P = 8
    stack = train_positional(TRAIN, P)
    assert torch.equal(stack[:2], torch.zeros(2, H))
    assert torch.allclose(stack[2:], unit(EXPECTED_DIFF), atol=1e-6)


def test_inference_lands_each_vector_on_its_training_column():
    # Prompt batch of width T = 7 > P = 4: slots 0..3 go on columns 3..6, columns 0..2
    # are untouched, and every vector is scaled by factor * max_act = 6.
    P, T = 4, 7
    stack = train_positional(TRAIN, P)
    delta = apply_positional(stack, T)
    assert torch.equal(delta[:, : T - P], torch.zeros(2, T - P, H))
    assert torch.allclose(delta[:, T - P:], 6.0 * stack.expand(2, P, H), atol=1e-6)
    # k back from the end in training == k back from the end at inference
    for k in range(P):
        train_col = W - 1 - k
        d = delta[0, T - 1 - k]
        assert torch.isclose(d[1] / d[0], torch.tensor(float(train_col))), k


def test_inference_narrow_prompt_drops_deepest_slots():
    # Prompt narrower than num_positions (T = 3 < P = 4): the deepest slot is dropped,
    # and the last-token vector still lands on the last column.
    P, T = 4, 3
    stack = train_positional(TRAIN, P)
    delta = apply_positional(stack, T)
    assert torch.allclose(delta[0], 6.0 * stack[-T:], atol=1e-6)
    assert torch.isclose(delta[0, -1, 1] / delta[0, -1, 0], torch.tensor(5.0))


def test_decode_step_is_untouched():
    stack = train_positional(TRAIN, 4)
    assert torch.equal(apply_positional(stack, 1), torch.zeros(2, 1, H))


if __name__ == "__main__":
    stack = train_positional(TRAIN, 4)
    print("trained stack (num_positions=4), slot j <- training column W-4+j:")
    for j, v in enumerate(stack):
        print(f"  slot {j}  column {W - 4 + j}  {[round(x, 3) for x in v.tolist()]}  dim1/dim0 = {v[1] / v[0]:.1f}")
    delta = apply_positional(stack, 7)
    print("inference delta, prompt width T=7, factor*max_act=6:")
    for c in range(7):
        v = delta[0, c]
        src = f"from training column {v[1] / v[0]:.0f}" if v.abs().sum() > 0 else "untouched"
        print(f"  column {c}  {[round(x, 3) for x in v.tolist()]}  {src}")

    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print(f"all {len(tests)} checks passed")
