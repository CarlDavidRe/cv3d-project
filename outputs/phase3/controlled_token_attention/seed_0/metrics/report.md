# Phase 3 controlled comparison

Coverage target: `vis_a`.

## Held-out fixed-history results

| Policy | Samples | Huber | Regret | Spearman | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vggt_independent_history | 12000 | 0.008110 | 0.175766 | 0.706572 | 0.848879 |
| vggt_joint_token_attention | 12000 | 0.008938 | 0.151308 | 0.744132 | 0.873130 |

## Closed-loop results

| Policy | Objects | Coverage AUC | Final coverage | Regret | Spearman | NDCG@5 | Final Chamfer ↓ | Chamfer AUC ↓ | Final F@10pct ↑ | Median policy ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vggt_independent_history | 300 | 6.783379 | 0.864034 | 0.403826 | 0.373068 | 0.667705 | 0.198837 | 4.356356 | 0.558329 | 1.563 |
| vggt_joint_token_attention | 300 | 6.832137 | 0.867432 | 0.353659 | 0.416071 | 0.701699 | 0.223208 | 4.721996 | 0.492365 | 244.261 |

## Conclusion

The expressive joint variant improves a clear majority of the reported outcomes.

Phase 2 random, farthest, PUN, VGGT, and oracle results remain external references. Oracle is a privileged upper bound; learned references retain their original NUM supervision.
