# DistilBERTClassifer on 4-way News Topic Classification
## Model Details
 * **Model Name:** DistilBERTClassifer.
 * **Base Model:** `distilbert-base-uncased` from [HuggingFace](https://huggingface.co/distilbert/distilbert-base-uncased).
 * **Developers:** Teun Boersma (s5195179), Julian Sprietsma (s5096219), Marcus Harald Olof Persson (s5343798).
 * **Model Type:** Transformer-based sequence classifier.
 * **Task:** Multi-class text classification (4 classes).
 * **Language:** English.
## Intended Usage and Limitations
This model's primary usage is for the classification of news articles into four specific classes: World, Sports, Business, and Sci/Tech. Regarding scope, the model is specifically finetuned on this dataset and may not generalise to non-news domains or different language contexts without further finetuning.
## Training Data
The dataset originates from the [AGNews dataset](https://huggingface.co/datasets/sh0416/ag_news). For preprocessing, the model utilises a combined "text" field created by concatenating the "title" and "description" columns from the dataset. With labelling of data:
 * 1: World.
 * 2: Sports.
 * 3: Business.
 * 4: Sci/Tech.
## Training Procedure
The model was trained using a custom `Trainer` class on the University of Groningen's high-performance computing cluster Hábrók.
 * **Hardware:** 1x NVIDIA A100 GPU, 4 CPUs, 16GB RAM.
 * **Optimiser:** AdamW with a learning rate of $2 \times 10^{-5}$.
 * **Loss Function:** `CrossEntropyLoss`.
 * **Batch Size:** 32.
 * **Early Stopping:** Triggered after 4 epochs without improvements, where the best model weights being restored from epoch 3.

| Epoch | Train Loss | Eval/Valid Loss | Note |
| - | - | - | - |
| 1 | 0.2352 | 0.2636 | |
| 2 | 0.1410 | 0.2690 | |
| 3 | 0.1011 | 0.2356 | Best Model |
| 4 | 0.0695 | 0.2955 | |
| 5 | 0.0478 | 0.3328 | |
| 6 | 0.0331 | 0.3674 | Early stopping triggered |
## Results
|  | Dev Set | Test Set |
| - | - | - |
| Accuracy | 0.9226 | 0.9424 |
F1 Score (Weighted) | 0.9224 | 0.9424 |

| Predicted \ Actual | World | Sports | Business | Sci/Tech |
| - | - | - | - | - |
World | 1785 | 14 | 57 | 44 |
Sports | 10 | 1880 | 6 | 4 |
Business | 32 | 8 | 1724 | 136 |
Sci/Tech | 26 | 8 | 93 | 1773 |
## Biases and Risks
 * The model inherits potential biases from the `distilbert-base-uncased` pretraining data.
 * Performance may drop significantly if applied to news data significantly newer than the training set.
 * The model most frequently confuses "Business" news with "Sci/Tech".
 ## Environmental Impact
 * **Data Centre:** Hábrók is maintained at the Coenraad Bron Center, which is [reported to operate at a Power Usage Effective (PUE) of 1.25 or below](https://www.rug.nl/groundbreakingwork/projects/datacenter/design-en-duurzaamheid).
 * **Location:** Groningen, the Netherlands.
 * **Total Runtime:** Approximately 51 minutes and 52 seconds, rounded up to an hour.
 * **Carbon Emitted:** From the [Machine Learning Emissions Calculator](https://mlco2.github.io/impact), assuming the carbon effiency is OECD's 2014 yearly average ($0.432 \text{ kg}/\text{kWh}$), then  $0.11 \times 1.25 = 0.1375 \text{ kg CO}_2$ was approximately emitted.