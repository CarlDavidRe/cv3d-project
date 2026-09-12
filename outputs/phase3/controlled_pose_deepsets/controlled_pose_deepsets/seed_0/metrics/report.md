# Phase 3 controlled comparison

Coverage target: `vis_a`.

## Held-out fixed-history results

| Policy | Samples | Huber | Regret | Spearman | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vggt_independent_history | 12000 | 0.008110 | 0.175766 | 0.706572 | 0.848879 |
| vggt_joint_pose_deepsets | 12000 | 0.008858 | 0.161828 | 0.719609 | 0.861223 |

## Closed-loop results

| Policy | Objects | Coverage AUC | Final coverage | Regret | Spearman | NDCG@5 | Final Chamfer ↓ | Chamfer AUC ↓ | Final F@10pct ↑ | Median policy ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vggt_independent_history | 300 | 6.783379 | 0.864034 | 0.403826 | 0.373068 | 0.667705 | 0.198837 | 4.356356 | 0.558329 | 1.775 |
| vggt_joint_pose_deepsets | 300 | 6.816212 | 0.864770 | 0.385708 | 0.370188 | 0.673958 | 0.175206 | 4.361906 | 0.566984 | 244.721 |

## Conclusion

The expressive joint variant improves a clear majority of the reported outcomes.

Phase 2 random, farthest, PUN, VGGT, and oracle results remain external references. Oracle is a privileged upper bound; learned references retain their original NUM supervision.
