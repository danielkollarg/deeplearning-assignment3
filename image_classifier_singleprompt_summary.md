# image_classifier_singleprompt — Conversation Summary

This document summarizes the conversation that produced
`image_classifier_singleprompt.ipynb`.

## Request

Build a FashionMNIST image classifier consolidated into a single Jupyter
notebook whose cells run sequentially, named `image_classifier_singleprompt`.
The notebook was to be generated from the prompt alone (an existing
`image_classifier.ipynb` in the repo was explicitly **not** used as a
reference). The code only needed to be correct and compile cleanly — running it
end-to-end on the backend was out of scope.

## What was built

`image_classifier_singleprompt.ipynb` — 15 cells (markdown + code) in run order:

1. **Load data** — FashionMNIST via TorchVision; a 60,000-image train/validation
   dataset and a 10,000-image test dataset, with the train/validation data split
   into 55,000 train / 5,000 validation using a seed-42 generator.
2. **DataLoaders** — train (shuffled with a seeded generator), validation, and
   test loaders, all with batch size 32.
3. **Sample data** — inspects the first sample's shape (`(1, 28, 28)`) and dtype
   (`torch.float32`) and lists the 10 class names.
4. **Model** — `ImageClassifier`, an MLP with flexible `input_size`,
   `hidden_sizes`, and `num_classes` parameters that assembles an
   `nn.Sequential` (Flatten → Linear/ReLU blocks → output Linear); hidden layers
   `[300, 100]`; loss function `nn.CrossEntropyLoss`.
5. **Optimizer + training** — SGD (`lr=0.01`) and a `train()` function that runs
   20 epochs, reports per-epoch train/validation loss and accuracy using
   torchmetrics `MulticlassAccuracy(average="micro")`, and returns a metric
   `history` dict.
6. **Evaluation** — runs over the validation loader, takes `argmax` of the
   logits, and compares to the labels to identify correct predictions.
7. **Softmax + model size** — applies `F.softmax` to a sample's logits, uses
   `torch.topk(probs, 4)` for the top-4 values/indices, and counts total
   parameters with `numel()` (266,610).

## Verification

- The notebook is valid JSON in `nbformat` 4.5.
- Every code cell was parsed with Python's `ast` module and compiles without
  error.
- The parameter count was confirmed by hand:
  `784·300 + 300` (235,500) `+ 300·100 + 100` (30,100) `+ 100·10 + 10` (1,010)
  `= 266,610`, matching the expected total.
- `torch`/`torchvision`/`torchmetrics` are not installed in the build
  environment, so the notebook was not executed — this was in line with the
  request (compile-clean, no runtime errors expected).

## Notes

- The notebook was written from the prompt only, per the instruction not to
  reference the existing `image_classifier.ipynb`.
- Work was committed and pushed to branch `claude/dreamy-mccarthy-wwgcor`.

## Follow-up fix: `ModuleNotFoundError: No module named 'torchmetrics'`

Running the notebook raised `ModuleNotFoundError: No module named 'torchmetrics'`
because `torchmetrics` was not installed in the kernel. A `## 0. Setup` cell was
added at the top of the notebook that runs `%pip install -q torch torchvision
torchmetrics`, installing the dependencies into the running kernel so the
notebook is self-contained. This fix was merged directly into the default
branch.

## Follow-up: plot the training accuracy

A `## 8. Plot the training accuracy` section was added at the end of the
notebook. It uses matplotlib to plot the per-epoch training accuracy from the
`history` dict returned by `train()`, overlaying the validation accuracy for
comparison. `matplotlib` was added to the setup `%pip install` cell (and is
already listed in `requirements.txt`).
