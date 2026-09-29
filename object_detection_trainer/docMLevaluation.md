# Evaluation Metrics in Machine Learning of the AI Trainer ONNX Evaluation

## 1. Accuracy
**Formula:** 
\[
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}
\]
**Description:** This measures the proportion of correct predictions (both true positives and true negatives) out of all predictions.

## 2. Precision
**Formula:** 
\[
\text{Precision} = \frac{TP}{TP + FP}
\]
**Description:** This measures the proportion of true positives out of all positive predictions made by the model.

## 3. Recall
**Formula:** 
\[
\text{Recall} = \frac{TP}{TP + FN}
\]
**Description:** This measures the proportion of true positives out of all actual positive cases.

## 4. Confusion Matrix
**Description:** The confusion matrix provides a comprehensive view of the performance, showing the counts of true positives (TP), false negatives (FN), false positives (FP), and true negatives (TN). It helps in understanding the breakdown of correct and incorrect predictions.

### Table Representation:

|                | **Ground Truth Positive** | **Ground Truth Negative** |
|----------------|---------------------------|---------------------------|
| **Predicted Positive** | TP                        | FP                        |
| **Predicted Negative** | FN                        | TN                        |

- **TP (True Positive):** When the model correctly predicts a positive case.
- **FP (False Positive):** When the model predicts a positive case incorrectly.
- **FN (False Negative):** When the model fails to predict a positive case that is present.
- **TN (True Negative):** Correctly predicting the absence of a case.


## 5. Intersection over Union (IoU)
**Formula:**
\[
\text{IoU} = \frac{BB_{\text{pred}} \cap BB_{\text{gt}}}{BB_{\text{pred}} \cup BB_{\text{gt}}}
\]
**Description:** IoU is used to measure the overlap between the predicted bounding box \(BB_{\text{pred}}\) and the ground truth bounding box \(BB_{\text{gt}}\). A higher IoU indicates a better prediction.

### Application in Object Detection:
- For each detected object, the IoU metric is used to determine if the detection is a true positive (TP).
- The confusion matrix is updated based on the matches between detected bounding boxes and ground truth boxes.
- Metrics such as accuracy, precision, and recall are calculated using the values from the confusion matrix.

These metrics are crucial in evaluating how well the object detection model is performing, guiding further training and tuning processes.

---

**Author:** 
```
Jimmy Majumder 
Sr. Robotics Engineer
Dept. of Development, 
QiBiTech Inc.
```
