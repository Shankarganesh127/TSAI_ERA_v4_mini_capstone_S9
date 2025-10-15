
---

# Project requirements — step-by-step

## Phase 0 — Planning & repo setup (Do this immediately)

1. Create a **public GitHub repository** named something like `resnet50-imagenet-capstone`.

   * Add collaborators (your partner if group).
   * Protect `main` branch (optional).
   * Create these top-level files/folders:

     ```
     README.md
     train/                # training scripts (train.py or wrapper)
     configs/              # example config files
     datasets/             # dataset download/prep scripts (imagenet -> expected layout)
     logs.md               # epoch-by-epoch logs (empty template)
     ec2_screenshots/      # put EC2 screenshot(s) here
     hf_space/             # HF Space app code (Gradio app)
     checkpoints/          # (or provide S3 link)
     experiments/          # notebooks / Colab quick runs
     LICENSE
     ```
2. Add `logs.md` initial template with header:

   ```
   # Training logs

   epoch | train_loss | train_top1 | train_top5 | val_loss | val_top1 | val_top5 | timestamp | notes
   ----- | ---------- | ---------- | ---------  | -------- | -------- | -------  | --------- | -----
   ```
3. Create a brief project timeline in `README.md` and assign ownership if you’re in a group (name + email).

**Deliverable produced:** GitHub repo link (used later in email to admin).

---

## Phase 1 — Quick prototype on small data (mandatory “dry-run”)

Purpose: validate code, data pipeline, augmentations, logging, checkpointing, and inference export before using EC2 credits.

4. Dataset: pick **Imagenette** or **Imagenet-mini** (small subset of ImageNet) or use ~1–5% of ImageNet if you already have it.

   * Implement `datasets/prepare_small.sh` to download and format dataset.
5. Implement `train/run_debug.sh` that runs one short training (e.g., 10 epochs) on Colab or a cheap GPU:

   * Show `nvidia-smi`, check GPU/AMP, check logs output to `logs.md`.
   * Save a checkpoint `checkpoints/debug_epoch10.pth`.
6. Record the results of this run inside `experiments/colab_debug.md` and append the per-epoch outputs to `logs.md`.

**Checkpoint (must have before emailing admin):**

* Working training command that runs to completion on small data.
* A `logs.md` containing those small-run epochs.
* A screenshot (or notebook cell output) proving it ran on Colab / local GPU.

---

## Phase 2 — Prepare full ImageNet pipeline & email the admin (deadline: **before 18th October**)

This is the gate to get TSAI credits.

7. Finalize your training code and make it fully trainable on ImageNet (no hard-coded paths).

   * Provide a single train command in README under **"Train on EC2"** with required env vars (DATA_DIR, OUTPUT_DIR).
8. Create the email to admin. **Email must include:**

   * Short subject: `Request for EC2 credits — ResNet50 ImageNet trainable pipeline`
   * Body that includes:

     * Short project description.
     * Link to your **GitHub repo** (must show the final code that will run on EC2).
     * What each team member will do (if group) and GitHub usernames/emails; point to commits that show both members have pushed.
     * State budget requested ($25 solo / $50 group) and that you will use provided credits for full ImageNet training.
     * Ask for the credit code and any limits/expiry they enforce.
   * Attach or link to the `logs.md` small-run proof and the Colab/EC2 debug screenshot.

**Deliverable produced:** Sent email (retain sent-email screenshot or copy).

---

## Phase 3 — EC2 setup & dry run on target instance

9. EC2 instance selection (based on credits):

   * For debugging: `g4dn.xlarge` (T4) — low cost.
   * For full training if credits allow: `p3.2xlarge` (V100) or multi-GPU p3/p4 if credits cover it.
10. EC2 checklist (run before full training):

* Attach EBS volume (fast SSD, >500GB for ImageNet).
* Install CUDA, cuDNN, PyTorch, NVIDIA drivers. Test:

  ```bash
  nvidia-smi
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
* Copy ImageNet onto instance (S3 -> EBS or attach EBS snapshot). Use a script `datasets/sync_to_ec2.sh`.

11. Dry run: launch a **single-epoch** training on full ImageNet (or a small sample) to verify multi-worker dataloader, checkpointing, and resume logic.
12. **Take the EC2 screenshot** required by the assignment:

* Terminal running the training command showing epoch progress OR `tail -f` of training logs.
* `nvidia-smi` output visible in separate terminal. Save as `ec2_screenshots/ec2_training_screenshot.png`.

**Deliverable produced:** `ec2_screenshots/ec2_training_screenshot.png` in repo.

---

## Phase 4 — Full training (ImageNet 1k) — run to target accuracy

13. Training recipe (exact requirements):

* Train **ResNet-50 from scratch** (do not use a pretrained model).
* Use the training recipe (example):

  ```
  epochs: 90
  batch-size: 256 (total)
  optimizer: SGD momentum 0.9
  lr: 0.1 * (batch_size / 256)
  weight_decay: 1e-4
  lr_schedule: cosine (or step [30,60,80])
  warmup_epochs: 5-10
  augmentations: RandomResizedCrop, RandomHorizontalFlip, RandAugment or AutoAugment, MixUp 0.2, CutMix 1.0
  amp: enabled
  checkpoint: save every N epochs (N<=5)
  ```

14. Logging format: append each epoch to `logs.md` in this format (one line per epoch):

```
epoch: <n>  train_loss: <x>  train_top1: <x%>  train_top5: <x%>  val_loss: <x>  val_top1: <x%>  val_top5: <x%>  timestamp: <ISO8601>
```

* Keep `logs.md` in the repo and push periodically so graders see progress.

15. Checkpointing & failure handling:

* Save model and optimizer state every 1–5 epochs.
* Use resume-from-checkpoint logic and cloud storage (S3) backup if using spot instances.

16. Target: **≥75% top-1 validation accuracy** (5000 bonus points if ≥81%). Stop when you reach the accuracy or run out of budget/credits.

* If training hits target early (say at epoch 70), continue to at least one more validation checkpoint to confirm.

**Deliverable produced:** `logs.md` containing epoch 1..N entries up to the stop point.

---

## Phase 5 — Export model + deploy HuggingFace Spaces app

17. Export model to CPU-friendly format for HF Spaces:

* Prefer options: `torch.jit.trace` (TorchScript), ONNX, or `transformers`-style convertors if you wrap your classifier.
* Example:

  ```python
  model.cpu().eval()
  example = torch.randn(1,3,224,224)
  traced = torch.jit.trace(model, example)
  torch.jit.save(traced, "model_cpu.pt")
  ```
* Optionally quantize or prune for faster CPU inference.

18. Build a small Gradio app in `hf_space/` that:

* Loads `model_cpu.pt` and runs `predict(image)` returning predicted label(s) + probabilities.
* Provide a demo page with example images and a simple UI.

19. Push `hf_space/` to GitHub and **deploy to HuggingFace Spaces** (public). Confirm it runs and note the Actions link.

* Include the HF Spaces link in your README and a small `hf_space_link.txt`.

**Deliverable produced:** HF Spaces app link (live) and model export files (or link to S3 if model > 500MB).

---

## Phase 6 — Documentation, YouTube demo & final submission

20. README: include all commands to reproduce (train, resume, export, deploy HF Space), links to checkpoints (or S3), and the `logs.md` pointer.
21. YouTube demo (3–7 minutes):

* Show training logs (scroll through `logs.md`), demonstrate HF Space classification, show EC2 screenshot, and explain your recipe and why you reached the reported accuracy.
* Include the YouTube link in README and in your final submission materials.

22. Final submission package should include:

* GitHub README link (final code + `logs.md`).
* HF Spaces app link.
* YouTube demo link.
* EC2 training screenshot (in repo).
* For groups: state partner email and ensure both members have commits visible.

---

## Phase 7 — Email template & what to send admin (concrete text)

Use this template to request credits (edit fields in `<>`):

```
Subject: Request for EC2 credits — ResNet50 ImageNet trainable pipeline

Hello <Admin Name>,

We request EC2 credits to run a ResNet-50 from-scratch training on ImageNet 1k as part of the TSAI capstone project.

GitHub repo (trainable pipeline): <https://github.com/your-org/resnet50-imagenet-capstone>
Proof of a working small-run and logs: <link to logs.md or experiment file>
Team members: <Name1> (<email1>), <Name2> (<email2>) — both have pushed commits shown at <commit link>

Planned usage:
- Instance: p3.2xlarge (or equivalent) for full training.
- Budget requested: $25 (solo) / $50 (group).
- Short plan: run a validation dry-run, then full training with checkpointing and spot instance fallback.

Please let us know the credit code and any restrictions. We will send you the final training logs, HuggingFace Space link, and YouTube demo once training completes.

Thanks,
<Name(s)>
```

---

## Risk mitigation & budget control (practical)

* Use **Colab / local** for prototyping to avoid wasting credits.
* Use **spot** instances for full training with frequent checkpointing.
* Use **mixed precision** and high-efficiency dataloading.
* If credits are denied, run a pared-down version (fewer epochs + aggressive augmentations) — **note:** hitting 75% without full training is unlikely.
* If running alone, budget plan: allocate ~$10–15 to test runs and save credits for a final run; credits should cover the heavy lifting.

---

## Acceptance checklist (what your deliverable MUST contain)

* [ ] Public GitHub repo with full training code and `logs.md`.
* [ ] `logs.md` contains epoch-by-epoch logs from epoch 1 to final epoch.
* [ ] EC2 screenshot showing you used EC2 for training.
* [ ] Live HuggingFace Spaces app link (Space demonstrates CPU inference).
* [ ] YouTube demo link (shows EC2, logs, and the HF Space in action).
* [ ] If group: partner email listed in submission and both partners have pushed commits.
* [ ] Sent email to admin before **18th October** (attach copy or note).

---

## Quick reference — sample logs line (copy-paste)

Put these lines into `logs.md` as you train (one per epoch):

```
epoch: 1  train_loss: 4.512  train_top1: 1.2%  train_top5: 5.8%  val_loss: 4.389  val_top1: 1.5%  val_top5: 6.2%  timestamp: 2025-10-11T14:02:13Z
...
epoch: 90 train_loss: 0.812  train_top1: 76.12% train_top5: 92.30% val_loss: 0.923  val_top1: 75.01% val_top5: 91.8% timestamp: 2025-11-01T03:22:10Z
```

---

If you want, I can now:

* scaffold the GitHub repo skeleton (train script + `logs.md` template + README), **or**
* produce the exact `train.py` + `requirements.txt` based on **timm** or **NVIDIA resnet** (you previously chose between them) so you can immediately test on Colab.

Which do you want next: repo skeleton (recommended) or a ready-to-run training script (timm or NVIDIA resnet)?
