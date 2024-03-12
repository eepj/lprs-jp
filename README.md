# **Japanese License Plate Recognition**
![Example use case](./images/AYapiMKfSYI.png)
<br>
Intended for research purposes only.

## **Web Demo**
Check out the web demo at [https://huggingface.co/spaces/eepj/jlpr](https://huggingface.co/spaces/eepj/jlpr).

You can run the model on the example images or on your own images.
The vehicles must be partially visible for the license plates to be detected. As of now the model can only identify license plate numbers if they are appropriately angled.

## **Feature Roadmap**
### **License Plate Detection**
✅ Single license plate detection <br>
✅ Multiple license plate detection <br>

### **Character Recognition**
✅ Identification number recognition <br>
✅ Full license plate recognition <br>

### **Suppported License Plate Types**
✅ Standard license plates<br>
✅ Commemorative license plates<br>
✅ Character-glowing license plates<br>

### **To Be Implemented**
❎ License plate perspective unwarping<br>
 
## **Approach**
### **License Plate Localization**
The **License Plates Dataset** comprises images of various vehicles annotated with their corresponding license plate bounding boxes. The dataset comes with 350 annotated images of license plates in a 70-20-10 split ratio.

The license plate detection model was fine-tuned on a pre-trained YOLOv8 backbone. All localized license plates are resized to 128x64 pixels for further processing.

### **Character Recognition**
Most of the images used for training the character recognition models were available from the **alpr_jp** dataset. But if a region code was missing from **alpr_jp**, additional images from Google Search were used to supplement the dataset.

Whenever possible, a minimum of 10 images were gathered for each marking. As the occurence of some markings is relatively rare, the minimum number of these markings could not be guaranteed.

Characters were extracted from **alpr_jp** images with pre-defined ROIs and labeled manually. The labeled images was stratified in a 60-20-20 split ratio.

The character recognition problem is modeled as a multi-class classification problem. The model described in *Chinese License Plate Recognition System Based on Convolutional Neural Network* was implemented with PyTorch.

### **Data Augmentation**
A data augmentation pipeline was used to increase the diversity of training samples. Below is an example of an white-on-green region code generated from a green-on-white sample through the data augmentation pipeline:

![Ise-Shima](./images/HIMwhOP3XxY.png)

## **Training**
### **Hardware and Schedule**
The models were trained on an Apple 7-core GPU M1 processor with MPS hardware acceleration. Each model was trained for 100 epochs.

### **Optimizer and Learning Rate**
The models were trained using the Adam optimizer with cross entropy loss. The initial learning rate was set to 1e-3, and the learning rate is adjusted by a factor of 0.1 every 30 epochs.

## **Metrics**
### **License Plate Localization**
|Model|Precision(B)|Recall(B)|mAP50(B)|mAP50-95(B)|Fitness|
|-----|------------|---------|--------|-----------|-------|
|**Fine-tuned YOLOv8**|0.84940|0.75702|0.85128|0.63573|0.65729|

### **Character Recognition**
|Model|Convolutional<br>Layers|Samples<br>(Classes)|Accuracy|Weighted<br>F1|Params<br>(10<sup>3</sup>)||
|-----|------------|-------------|--------|-----------|-----------------------|-|
|**Region Code**<br><br><br>|32, 64, 128<br>32, 64, 128, 256<br>16, 32, 64, 128|412 (134)<br><br><br>|0.93046<br>0.97816<br>0.97330|0.92476<br>0.97543<br>0.97289|368<br>462<br>132|<br>✅<br>(a)|
|**Vehicle Class Code**<br><br><br>|32, 64, 128<br>16, 32, 64<br><br>|444 (11)<br><br><br>|0.97478<br>0.98423<br><br>|0.97760<br>0.98298<br><br>|97.9<br>25.9<br><br>|(b)<br>✅<br><br>|
|**Hiragana Code**<br><br><br>|32, 64, 128<br>32, 64, 128, 256<br>16, 32, 64, 128|430 (43)<br><br><br>|0.95814<br>0.97907<br>0.97674|0.95581<br>0.97776<br>0.97519|143<br>400<br>103|<br>✅<br>(a)|
|**Identification Number**<br><br><br>|32, 64, 128<br>32, 64, 128, 256<br>16, 32, 64|547 (11)<br><br><br>|0.99086<br>0.99269<br>0.99086|0.99092<br>0.99271<br>0.99086|104<br>395<br>29.4|<br>(b)<br>✅|

## **Observations**
* (a) Model not deployed despite comparable metrics as a substantial decrease in confidence in the predicted classes was observed.

* (b) "*The bigger the better*" does not necessarily hold true. In some cases, increasing the number of parameters resulted in marginal improvement or degraded performance. The complexity of the problem should be taken into consideration.

* Degraded recognition performance on minority class markings is observed. This is likely due to insufficient training samples available for these classes.

## **References**
**alpr_jp** <br>
Huge thanks to dyama-san for sharing the dataset. <br>
https://github.com/dyama/alpr_jp

**License Plates Dataset** <br>
https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn

**Chinese License Plate Recognition System Based on Convolutional Neural Network** <br>
H. Chen, Y. Lin, and T. Zhao, ‘Chinese License Plate Recognition System Based on Convolutional Neural Network’, Highlights in Science, Engineering and Technology, vol. 34, pp. 95–102, 2023. <br>
https://www.researchgate.net/publication/369470024

## **Fun Fact**
This repository was created on [Leap Day 2024](https://doodles.google/doodle/leap-day-2024/).