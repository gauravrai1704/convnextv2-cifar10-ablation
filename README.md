# ConvNeXt V2 on CIFAR-10: A Reproducibility & Ablation Study

> Reproducing and ablating key design decisions from  
> **Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with  
> Masked Autoencoders", NeurIPS 2023**  

---

## Motivation

ConvNeXt V2 introduces Global Response Normalization (GRN) to prevent  
feature collapse during masked autoencoder pretraining. This project asks:  
**does GRN help in purely supervised training too?**  
We answer this — and more — through 8 controlled experiments on CIFAR-10.

---

## Results

| Experiment | Params | Test Acc |
|---|---|---|
| E1: Vanilla CNN | 0.23M | 81.85% |
| E2: ResNet-18 | 11.17M | 93.86% |
| E3: ConvNeXt V1 | 7.40M | 71.89% |
| E4: ConvNeXt V2 (Full) | 7.40M | 76.23% |
| E5: V2 + LayerScale, no GRN | 7.40M | 73.66% |
| **E6: V2 – No Stochastic Depth** | **7.40M** | **77.05%**  best scratch |
| E7: V2 – No GRN | 7.40M | 73.15% |
| **E8: Pretrained V2 (timm)** | **28.0M** | **97.56%**  best overall |

---

## Key Findings

1. **GRN is the most important V2 component** — removing it (E7) causes  
   the largest accuracy drop (−3.08% vs full V2), confirming its role  
   as a channel diversity regularizer even without masked autoencoder pretraining.

2. **Stochastic Depth hurts at CIFAR-10 scale** — E6 (no Stochastic Depth)  
   outperforms full V2 by 0.82%, suggesting this regularizer is less effective  
   for shallow networks on small datasets.

3. **ResNet-18 dominates all scratch-trained ConvNeXt variants** — a scale  
   mismatch effect: ConvNeXt's 7×7 kernels are designed for 224×224 ImageNet  
   images, not 32×32 CIFAR-10.

4. **Transfer learning gap is enormous** — pretrained V2 achieves 97.56%  
   in a single fine-tuning epoch vs 77.05% best scratch result.

---


## Setup & Reproducibility

All experiments use identical settings:
- **Dataset**: CIFAR-10, split 45k / 5k / 10k (train / val / test), seed=42
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.05)
- **Schedule**: Cosine annealing, 50 epochs, min_lr=1e-6
- **Augmentation**: RandAugment + RandomCrop + RandomErasing
- **Loss**: Cross-entropy with label smoothing 0.1
```bash
pip install torch torchvision timm matplotlib pandas seaborn
```

Ran in Google Colab with T4 GPU.  

---


## Reference
```bibtex
@inproceedings{woo2023convnextv2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei 
          and Liu, Zhuang and Kweon, In So and Xie, Saining},
  booktitle={NeurIPS},
  year={2023}
}
```


