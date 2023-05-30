import torch
import torch.nn as nn

def binary_accuracy(preds, y):
    # apply sigmoid
    preds = nn.functional.sigmoid(preds)
    # apply threshold of 0.5
    one_preds = preds > .5
    # get accuracy
    accuracy = (one_preds == y).sum() / len(preds)
    return accuracy

def multilabel_accuracy(logits, y):
        #logits is (batchsize, num_labels), y is (batchsize,)
    logits = nn.functional.softmax(logits)
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == y).sum() / len(preds)
    return accuracy