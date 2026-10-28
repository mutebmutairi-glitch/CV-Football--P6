# CV-Football (Project 6): Phase 1

AI251P Applied Computer Vision · KAUST Academy · Project 6: Player and Ball Detection in Football Video

Phase 1 covers data preparation and player-role classification (player / goalkeeper / referee) on SoccerNet GSR 2025.

## Links
| Item | Link |
|---|---|
| Notebook (code + outputs) | [`football_phase1_Final.ipynb`](football_phase1_Final.ipynb) |
| Dataset card | [`dataset_card.md`](dataset_card.md) |
| Model card (acceptance criteria) | [`model_card.md`](model_card.md) |
| W&B project | https://wandb.ai/muteb-mutairi-kaust/cv-football-phase1-kaggle |
| Data source | https://huggingface.co/datasets/SoccerNet/SN-GSR-2025 |

## Headline result (held-out test match, 1,763 crops)
Frozen ConvNeXt-Tiny (torchvision) with a goalkeeper decision weight of 2.70 (chosen on validation): accuracy 91.09%, macro-F1 85.05%.

## How to reproduce
1. Open the notebook in Colab or Kaggle with a GPU.
2. Add a `WANDB_API_KEY` secret (Colab: key icon; Kaggle: Add-ons → Secrets).
3. Run all cells. The notebook downloads the data, builds the crops, splits by match (no leakage), trains with fixed seeds, evaluates the test match once and logs the runs to W&B in the last cells.

Data, crops and model weights are not committed (see `.gitignore`).

## Team
| Member | ID | Workstream |
|---|---|---|
| Faisal Al-Mutairi | 225472 | Data and classification |
| Ali Aljuaid | 225491 | Detection (Phase 2) |
| Fahad Alqahtani | 225496 | Segmentation (Phase 2) |
| Muteb Al-Mutairi | 225497 | Integration and reproducibility |
