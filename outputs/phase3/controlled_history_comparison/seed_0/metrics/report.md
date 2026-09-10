# Phase 3 controlled comparison

Coverage target: `vis_a`.

## Held-out fixed-history results

| Policy | Samples | Huber | Regret | Spearman | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vggt_independent_history | 12000 | 0.008110 | 0.175766 | 0.706572 | 0.848879 |
| vggt_joint_history | 12000 | 0.007683 | 0.198544 | 0.661221 | 0.826619 |

## Closed-loop results

| Policy | Objects | Coverage AUC | Final coverage | Regret | Spearman | NDCG@5 | Median policy ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vggt_independent_history | 300 | 6.783379 | 0.864034 | 0.403826 | 0.373068 | 0.667705 | 1.563 |
| vggt_joint_history | 300 | 6.753729 | 0.862319 | 0.417516 | 0.336778 | 0.651358 | 242.346 |

## Conclusion

Joint processing does not improve a clear majority of the prespecified outcomes.

Official PUN and Phase 2 VGGT remain external references; they retain original NUM supervision.
