# LOSSES

@source https://stackoverflow.com/questions/75809415/yolo-v8-result-grap

Losses These are components of the overall loss value:

- "**box_loss**" -> This is the bounding box regression loss, which measures the error
in the predicted bounding box coordinates and dimensions compared to the ground
truth. A lower box_loss means that the predicted bounding boxes are more
accurate.

- "**cls_loss**" -> This is the classification loss, which measures the error in the
predicted class probabilities for each object in the image compared to the
ground truth. A lower cls_loss means that the model is more accurately
predicting the class of the objects.

- "**dfl_loss**" -> This is the deformable convolution layer loss, a new addition to the
YOLO architecture in YOLOv8. This loss measures the error in the deformable
convolution layers, which are designed to improve the model's ability to detect
objects with various scales and aspect ratios. A lower dfl_loss indicates that
the model is better at handling object deformations and variations in
appearance.

The overall loss value is typically a weighted sum of these individual losses.
The specific units of the vertical axis would be dependent on the
implementation, but generally, they represent the magnitude of the error or the
difference between the predicted and ground truth values.

# EVALUATION METRICS

- **Precision** -> How many of the detected crow traps are actually crow traps
- **Recall** -> How many crow traps did we succesfully detect out of all the crow traps out there?

    > Precision and recall are often a trade-off!

- **F1** -> An harmonic mean of precision and recall, like accuracy gives a good overview of the model. Contrary to accuracy, works well with unbalanced datasets.

- **AP** -> The area under the precision-recall curve.

- **mAP** -> The average AP across all classes. A good mAP indicates a model that's stable and consistent across difference confidence thresholds.
    - **mAP@0.5** -> mAP at IoU threshold of 0.5
    - **mAP@0.5-95** -> mAP across IoU thresholds from 0.5 to 0.95
