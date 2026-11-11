# CV-Football (Project 6): Phase 1

AI251P Applied Computer Vision · KAUST Academy · Project 6: Player and Ball Detection in Football Video

Phase 1 builds the data foundation and a **player-role classifier** (player / goalkeeper / referee) on SoccerNet GSR 2025. The role label feeds Phase 2, which detects people and the ball, assigns players to teams and maps them onto a top-down pitch view.

## Team
| Member | ID | Workstream |
|---|---|---|
| Faisal Al-Mutairi | 225472 | Data and classification |
| Ali Aljuaid | 225491 | Detection (Phase 2) |
| Fahad Alqahtani | 225469 | Segmentation (Phase 2) |
| Muteb Al-Mutairi | 225497 | Integration and reproducibility |

## Links
| Item | Link |
|---|---|
| Main notebook (revised run, all report numbers) | [`football_phase1_Final_v2.ipynb`](football_phase1_Final_v2.ipynb) |
| Original run (25 Oct, first submission) | [`football_phase1_Final.ipynb`](football_phase1_Final.ipynb) |
| Pinned package versions | [`requirements-lock.txt`](requirements-lock.txt) |
| Dataset card | [`dataset_card.md`](dataset_card.md) |
| Model card (acceptance criteria) | [`model_card.md`](model_card.md) |
| W&B project | https://wandb.ai/muteb-mutairi-kaust/CV-Football--P6 |
| Data source | https://huggingface.co/datasets/SoccerNet/SN-GSR-2025 |

## Headline result
Held-out test match, 1,763 crops, 224×224 input:

**Frozen ConvNeXt-Tiny + goalkeeper weight ×2.70 (chosen on validation): accuracy 91.09% (95% CI 89.67–92.34), macro-F1 85.05%.**

## 1. Task
- **Input → output:** RGB person crop resized to 224×224 → role (player, goalkeeper, referee) with class probabilities.
- **Why role and not team A/B:** SoccerNet's `left`/`right` label is the pitch side inside one match, not a kit identity, so it cannot be learned across matches. Teams are separated per match by jersey colour instead.
- **Acceptance criteria** were fixed on 11 Oct 2026, before any result (see `model_card.md`).

## 2. Data (M1)
| Step | Result |
|---|---|
| Source | SN-GSR-2025 train: 57 sequences of 30 s, 48 processed, 3 independent matches (`game_id` 4, 6, 9) |
| Sampling | 1 frame per second (every 25th frame), max 35 per sequence → 1,440 frames, 23,386 person boxes |
| Cleaning | 0 corrupt frames, 0 invalid boxes, 1,592 tiny crops removed (< 24 px), 0 exact duplicates (SHA-256 over 21,794 unique crops); near-duplicate check: 15 pairs, none leaking between splits |
| Final dataset | 7,752 role crops after per-class caps |
| Label audit | 200 random crops checked by hand in the notebook: 0 errors, 95% upper bound 1.83% |

**Leakage-safe split:** one whole match per split (seed 42); an assertion checks that no match appears in two splits.

| Split | Crops | Player / goalkeeper / referee |
|---|---|---|
| Train | 4,138 | 3,500 / 194 / 444 |
| Validation | 1,851 | 875 / 323 / 653 |
| Test | 1,763 | 875 / 230 / 658 |

## 3. Classification (M2–M3)
Five ImageNet backbones were screened on validation; ConvNeXt-Tiny ranked first. Three runs were then trained (AdamW, class-weighted cross-entropy, early stopping on validation loss). The frozen model is selected on validation macro-F1. Note: this between-run rule was fixed after the first test results were seen; the notebook's original validation-loss ranking would pick the fully fine-tuned run (test accuracy 77.99%).

| Model (test match, 224 px, argmax) | Accuracy | 95% CI (Wilson) | Top-2 | Macro-F1 | ECE |
|---|---|---|---|---|---|
| Scratch CNN | 53.03% | 50.70–55.36 | 73.85% | 49.48% | 0.383 |
| Frozen ConvNeXt-Tiny | 89.96% | 88.47–91.28 | 96.88% | 80.47% | 0.022 |
| Fully fine-tuned ConvNeXt-Tiny | 77.99% | 76.00–79.86 | 99.32% | 74.25% | 0.189 |
| **Frozen + goalkeeper ×2.70** | **91.09%** | 89.67–92.34 | 98.13% | **85.05%** | 0.016 |

**Main findings**
- Referees are recognised perfectly (658/658).
- The main error is goalkeepers predicted as players (116 of 230), because training has only 194 goalkeeper crops from one match.
- Full fine-tuning overfits the single training match (validation loss jumps from 0.21 to above 2.0) and is over-confident; the frozen backbone generalises best and is the best calibrated.
- Multiplying the goalkeeper score by 2.70 (tuned on validation, no retraining) raises goalkeeper recall from 40.0% to 60.4% and macro-F1 from 80.5% to 85.1%.
- Team split by torso colour (k-means per match) reaches only 52.1% on the test match; improving it is a Phase 2 task.

**Next data steps:** more independent matches (especially goalkeepers and varied kits), motion-blur augmentation, and more matches per split for a match-level confidence interval.

## 4. Reproducibility
- One `CFG` dictionary holds every setting and is saved as `config.json` with each run.
- Seed 42 for Python, NumPy, PyTorch and CUDA.
- Exact package versions in [`requirements-lock.txt`](requirements-lock.txt) (committed); see also `requirements.txt`.
- Runs are tracked in W&B (link above).
- **Independent rerun:** the original run (25 Oct) and the revised run by a second team member on a separate Kaggle account (11 Nov, runs ending `20261111T171659Z` in W&B) give identical test results for the frozen and fully fine-tuned models; the scratch CNN differs by 0.05 points (GPU non-determinism).

## How to reproduce
1. Open `football_phase1_Final_v2.ipynb` in Kaggle with a GPU and internet enabled.
2. Run all cells. The notebook downloads the data, builds the crops, splits by match, trains with fixed seeds and evaluates the test match once.
3. At the label-audit cell, review the 200 crops and list any wrong labels in `WRONG`.

Data, crops and model weights are not committed (see `.gitignore`).
