| variant | backbone | feature | input_dim | trainable_parameters | best_epoch | huber_loss | official_unmasked_mse_loss | normalized_regret_mean | spearman_mean | ndcg_at_5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_mean_map | none | train_mean_map | 0 | 0 | 0 | 1.69965 |  | 0.259207 | 0.409823 | 0.754719 |
| raw_rgb_16x16_mlp | raw_rgb | flattened_rgb | 768 | 106160 | 29 | 1.42759 |  | 0.274919 | 0.401599 | 0.752508 |
| imagenet_vit_pooled_patch | imagenet_vit | pooled_patch | 768 | 106160 | 43 | 1.09042 |  | 0.234508 | 0.466335 | 0.782107 |
| imagenet_vit_cls_token | imagenet_vit | cls_token | 768 | 106160 | 20 | 1.22427 |  | 0.251905 | 0.447843 | 0.770926 |
| dinov2_pooled_patch | dinov2 | pooled_patch | 768 | 106160 | 32 | 1.14626 |  | 0.213018 | 0.493319 | 0.802052 |
| dinov2_cls_token | dinov2 | cls_token | 768 | 106160 | 14 | 1.22922 |  | 0.219228 | 0.492417 | 0.798307 |
| vggt_pooled_patch | vggt | pooled_patch | 2048 | 272560 | 50 | 1.151 |  | 0.228866 | 0.476902 | 0.788165 |
| vggt_max_pooled_patch | vggt | max_pooled_patch | 2048 | 272560 | 50 | 1.15173 |  | 0.211355 | 0.509235 | 0.806453 |
| vggt_camera_token | vggt | pooled_camera | 2048 | 272560 | 50 | 1.153 |  | 0.229236 | 0.495107 | 0.793186 |
| vggt_pooled_register | vggt | pooled_register | 2048 | 272560 | 50 | 1.15105 |  | 0.224569 | 0.486437 | 0.793918 |
| vggt_camera_patch | vggt | pooled_camera+pooled_patch | 4096 | 538800 | 50 | 1.11184 |  | 0.221118 | 0.497678 | 0.796789 |
| pun_upnet | pun_upnet | vit_small_patch16_224_official_pretrained | 384 | 21684144 |  | 0.848841 | 2.71683 | 0.169428 | 0.60185 | 0.846303 |
