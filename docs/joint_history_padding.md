# Joint-history padding convention

Phase 3 histories have different numbers of acquired views, but PyTorch batches
require tensors with one shared shape. The collator therefore pads every
history to the longest history in the current batch.

## Batch representation

Padding is always **suffix padding**: every real observation appears first and
is followed only by padded positions. The relevant tensors are:

- `history_images`: `[B, H_max, 3, image_height, image_width]`
- `history_anchor_ids`: `[B, H_max]`
- `history_padding_mask`: `[B, H_max]`

Real anchors use canonical IDs `0–47`. Padded anchors use `-1`. The padding mask
is `false` for a real observation and `true` for padding.

For example, histories with one and three observations are collated as:

```text
history_anchor_ids = [
    [ 4, -1, -1],
    [10, 22, 31],
]

history_padding_mask = [
    [false, true,  true ],
    [false, false, false],
]
```

The padded image slots contain zeros, but those zeros are storage placeholders;
they are not observations and must never be processed as images.

## Why ordinary padded forwarding is invalid

VGGT does not accept an observation-padding mask. Passing zero-filled padded
images into the model would therefore let fake views participate in global
attention. A padded black image could change the camera, register, and patch
tokens of real views, making the prediction depend on how the batch happened
to be padded. The same history could then produce different features when
batched beside a longer history.

Masking the features only after VGGT would be too late: cross-view interaction
would already have occurred inside the backbone.

## Length-grouped joint extraction

The joint extractor handles this before the VGGT call:

1. Validate that every sample contains at least one real observation.
2. Validate suffix padding. A pattern such as `[false, true, false]` is rejected
   because a real view appears after padding.
3. Compute each sample's real history length from the padding mask.
4. Group samples that have the same real history length.
5. Slice each group to that exact length, removing all padded image slots.
6. Run one frozen VGGT forward for the group with shape
   `[group_size, history_length, 3, H, W]`.
7. Pool the selected joint VGGT tokens into one history-conditioned feature per
   real view.
8. Restore those features to the original padded batch layout and fill padded
   feature positions with exact zeros.

In the example above, VGGT receives one forward of shape `[1, 1, 3, H, W]` and
one of shape `[1, 3, 3, H, W]`. It never receives a three-view sequence
containing two artificial black views.

Each individual history is still processed jointly: all real views in that
history share one VGGT sequence and can interact through the backbone's global
attention. Grouping only separates samples with incompatible sequence lengths;
it does not turn the real views into independent single-image forwards.

## Pooling after extraction

After joint extraction, the model appends the fixed canonical camera direction
to each real view feature and applies masked-mean history pooling. The same
padding mask excludes restored zero positions from both the sum and the view
count:

```text
pooled_history = sum(real_view_features) / number_of_real_views
```

This second mask is still necessary even though padded features are zero. If
padding were included in the denominator, a short history would be scaled down
whenever it shared a batch with a longer history.

## Enforced invariants

Tests enforce that:

- changing padded image values cannot change any output;
- padded feature positions are exactly zero;
- malformed non-suffix masks are rejected;
- another real view can change a frame's joint feature, confirming cross-view
  interaction;
- the joint head receives the same histories, anchor IDs, labels, and candidate
  masks as the independent control.

As a result, joint predictions can depend on the complete real observation
history, but cannot depend on artificial padding or on the maximum history
length of unrelated samples in the same minibatch.
