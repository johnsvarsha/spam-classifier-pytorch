# Spam Filter with a PyTorch Neural Network

This project builds a text classifier that reads a message and says if it is **spam** or **not spam (ham)**. It also tells you how confident it is.

## Files

- `spam_classifier_pytorch.ipynb` - full notebook with code, training curves and sample predictions
- `spam_model.pt` - the saved neural network
- `tfidf_vectorizer.joblib` - the saved text-to-numbers converter (needed to use the model)

## What I did

1. **Made a dataset** of 4,000 text messages, then removed repeated messages so the same message cannot be in both training and test data.
2. **Split the data first:** 70% train, 15% validation, 15% test.
3. **Turned text into numbers with TF-IDF.** It gives more weight to words that are special to a message. It learned its words from the training data only.
4. **Built a neural network in PyTorch** with 3 layers, **Batch Normalization** (steadier training) and **Dropout** (less overfitting).
5. **Trained with early stopping.** Training stops when the validation loss stops improving, and the best version of the model is restored.
6. **Plotted loss and accuracy curves.** Training loss keeps going down, but validation loss stops improving around epoch 20. That is the point where overfitting would start.
7. **Tested on unseen messages** and printed a confidence for each.

## Results (test set, 399 messages)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Neural network | 0.95 | 0.92 | 0.92 | 0.92 |
| Logistic Regression (simple baseline) | 0.95 | 0.92 | 0.92 | 0.92 |

The neural network found 115 of 125 spam messages and wrongly flagged 10 normal messages. It scored about the same as the simple baseline. On easy data, a deep network is not always better.

## Example predictions

| Message | Result | Confidence |
|---|---|---|
| WINNER! you have won a free holiday. call now to claim | SPAM | 97% |
| hey Priya are you coming to the cafe tonight | ham | 86% |
| the free trial of the app ends tomorrow, want to cancel it? | ham | 95% |

## Important notes

- The data is **made up (synthetic)**, because I had no internet to download a real dataset. About 3% of labels were flipped on purpose to make it harder. The scores show that the method works, not how it would do on real spam.
- Messages with words the model has never seen get unreliable confidence scores.
- Real spam changes over time, so a real filter needs regular retraining and human review of uncertain messages.
- To use a real dataset such as the UCI SMS Spam Collection, change only the data loading cell. It needs a `text` column and a `label` column (1 = spam, 0 = ham).

## How to run

```
pip install torch scikit-learn pandas numpy matplotlib joblib jupyter
jupyter notebook spam_classifier_pytorch.ipynb
```

Run all cells from top to bottom.

## How to use the saved model

```python
import torch, joblib
tfidf = joblib.load("tfidf_vectorizer.joblib")
# SpamNet is the class defined in the notebook
ck = torch.load("spam_model.pt")
model = SpamNet(ck["n_in"]); model.load_state_dict(ck["state_dict"]); model.eval()
X = torch.tensor(tfidf.transform(["you won a free prize, call now"]).toarray(), dtype=torch.float32)
print(torch.sigmoid(model(X)))   # probability of spam
```
