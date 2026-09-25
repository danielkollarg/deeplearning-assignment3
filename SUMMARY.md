# Project Summary

A PyTorch image classifier for the FashionMNIST dataset, built step-by-step and
stored in this repository (branch `claude/pytorch-image-classifier-6pi6i0`).

## Goal

Build a FashionMNIST image classifier, generated incrementally, with all work
consolidated into a single Jupyter notebook whose cells run sequentially.

## Notebook: `image_classifier.ipynb`

The work began as separate `.py` scripts and was later consolidated into one
notebook with cells that run in order:

0. **Setup** — install dependencies (`torch`, `torchvision`, `torchmetrics`)
   into the running kernel via `%pip` so the notebook is self-contained.
1. **Load data** — FashionMNIST via TorchVision; train/validation + test
   datasets, split 55,000 / 5,000 with seed 42.
2. **DataLoaders** — train (shuffled, seed 42), validation, and test loaders;
   all batch size 32.
3. **Sample data** — inspect the first sample's shape/dtype (`(1, 28, 28)`,
   `torch.float32`) and list the 10 class names.
4. **Model** — `ImageClassifier` (MLP) with flexible params (`input_size`,
   `hidden_sizes`, `num_classes`) building an `nn.Sequential`; hidden layers
   `[300, 100]`; loss function `CrossEntropyLoss`.
5. **Optimizer + training** — SGD (`lr=0.01`); a `train()` function running 20
   epochs, reporting train/validation loss and accuracy. Accuracy uses
   torchmetrics `MulticlassAccuracy` (`average="micro"`); `train()` returns a
   metric `history`.
6. **Evaluation** — run over the validation loader, `argmax` the logits, and
   compare to the labels to check which predictions were correct.
7. **Softmax + model size** — `F.softmax` on a sample's logits,
   `torch.topk(probs, 4)` for the top-4 values/indices, and the total parameter
   count via `numel()` (266,610).

## Repository housekeeping

- `.gitignore` for the downloaded `data/`, caches, and checkpoints.
- `requirements.txt`: `torch`, `torchvision`, `torchmetrics`.
- Each step was committed and pushed to the feature branch.
- Each cell was verified by parsing and unit-testing the logic on small or
  synthetic data.

## Notes

- The full 20-epoch training was not run end-to-end during development because
  each epoch is slow (minutes) on a CPU-only container — best run on a GPU
  runtime.
- Learning rate (`0.01`) and the micro accuracy averaging are defaults; adjust
  if the assignment specifies otherwise.
