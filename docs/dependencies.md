# Dependencies and checkpoints

Create the evaluator environments:

```bash
bash setup.sh
conda activate streamav
```

`setup.sh` creates `streamav` and `streamav-sync`.

## Checkpoints

Download the evaluator checkpoints:

```bash
bash scripts/download_assets.sh
```

Assets are stored in `.cache/assets` by default. Set
`STREAMAV_ASSETS_ROOT` to use another directory.

## Official sources

| Local path | Official source |
|---|---|
| `visual_quality/ViT-L-14.pt` | [OpenAI CLIP ViT-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) |
| `visual_quality/head.pth` | [LAION aesthetic predictor](https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth) |
| `audio_quality/model.pt` | [Audiobox Aesthetics](https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt) |
| `av_alignment/model.pth` | [ImageBind](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) |
| `av_synchronization/config.yaml` | [Synchformer configuration](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml) |
| `av_synchronization/model.pt` | [Synchformer checkpoint](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt) |
| `long-consistency/home/.cache/vbench/clip_model/ViT-B-32.pt` | [OpenAI CLIP ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
| `long-consistency/` | [DINO](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2), and [DreamSim](https://github.com/ssundaram21/dreamsim) |

Evaluator source is included under [`third_party/`](../third_party/). See
[`NOTICE`](../NOTICE) for upstream terms.
