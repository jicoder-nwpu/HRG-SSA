from io_utils import *
from sklearn import metrics
import os
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
from sklearn.metrics import confusion_matrix
import torch


from sklearn.metrics import accuracy_score

def class_accuracies(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    accuracies = np.diag(cm) / cm.sum(axis=1)
    
    return accuracies

def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = []

    for i in range(cm.shape[0]):
        sensitivity = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        per_class_accuracy.append(sensitivity)
    
    return np.mean(per_class_accuracy)

MELD_emotion_lables = {
    'neutral': 0, 
    'surprise': 1, 
    'fear': 2, 
    'sadness': 3, 
    'joy': 4,
    'disgust': 5, 
    'anger' : 6 
}

MELD_sentiment_labels = {
    'negative': 0, 
    'neutral': 1, 
    'positive': 2
}

IEMOCAP_emotion_lables = {
    'happiness': 0,
    'sadness': 1, 
    'neutral': 2, 
    'anger': 3, 
    'excitement': 4, 
    'frustration': 5
}

IEMOCAP_sentiment_labels = {
    'negative': 0, 
    'neutral': 1, 
    'positive': 2
}

def evaluate_metrics(res, dataset, logger=None):

    if not isinstance(res, dict) and os.path.exists(res):
        res = load_json(res)

    sen_pred = []
    sen_real = []
    emo_pred = []
    emo_real = []
    for dia in res:
        if dataset == 'meld':
            sen_pred.append(MELD_sentiment_labels[res[dia]['sentiment_pred']])
            sen_real.append(MELD_sentiment_labels[res[dia]['sentiment_label']])
            emo_pred.append(MELD_emotion_lables[res[dia]['emotion_pred']])
            emo_real.append(MELD_emotion_lables[res[dia]['emotion_label']])
        elif dataset == 'iemocap':
            sen_pred.append(IEMOCAP_sentiment_labels[res[dia]['sentiment_pred']])
            sen_real.append(IEMOCAP_sentiment_labels[res[dia]['sentiment_label']])
            emo_pred.append(IEMOCAP_emotion_lables[res[dia]['emotion_pred']])
            emo_real.append(IEMOCAP_emotion_lables[res[dia]['emotion_label']])

    emo_report = metrics.classification_report(emo_real, emo_pred, labels=list(range(len(MELD_emotion_lables))), target_names=MELD_emotion_lables.keys(), digits=4)
    emo_confusion = metrics.confusion_matrix(emo_real, emo_pred)

    print("Precision, Recall and F1-Score...")
    print(emo_report)

    print("emotion Confusion Matrix...")
    print(emo_confusion)
    

    f1 = f1_score(sen_real, sen_pred, average='micro' ) * 100
    p = precision_score(sen_real, sen_pred, average='weighted') * 100
    r = recall_score(sen_real, sen_pred, average='weighted') * 100

    sen_pred = torch.tensor(sen_pred)
    sen_real = torch.tensor(sen_real)
    correct = (sen_pred == sen_real).sum().item()
    total = sen_real.size(0)
    acc7 = correct / total

    emo_pred = torch.tensor(emo_pred)
    emo_real = torch.tensor(emo_real)
    correct = (emo_pred == emo_real).sum().item()
    total = emo_real.size(0)
    emo_acc7 = correct / total

    return str(acc7) + " " + str(emo_acc7)