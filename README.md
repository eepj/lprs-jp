# Japanese License Plate Recognition
![Example use case](./images/AYapiMKfSYI.png)

A Japanese license plate recognition project implemented with PyTorch.
> [!Note]
> For personal research purposes only. Project under development.

## Hugging Face Space Demo
Check out the demo at [https://huggingface.co/spaces/eepj/jlpr](https://huggingface.co/spaces/eepj/jlpr).

## Japanese License Plate Format
### Markings
<img src="./images/XZjptNTxOZF.png" height=150px>

① Name of Region Registered
<br>
② Classification Number
<br>
③ Kana Character
<br>
④ 4-Digit Designation Number (Leading zeros are shown as ．)

### Color Scheme
<table>
  <tr>
    <th>License Plate Type</th>
    <th>Engine Displacement</th>
    <th>Marking Color</th>
    <th>Background Color</th>
  </tr>
  <tr>
    <td>Private</td>
    <td>≥ 660 cc</td>
    <td>Green</td>
    <td>White</td>
  </tr>
  <tr>
    <td>Private</td>
    <td>< 660 cc</td>
    <td>Black</td>
    <td>Yellow</td>
  </tr>
  <tr>
    <td>Commercial</td>
    <td>≥ 660 cc</td>
    <td>White</td>
    <td>Green</td>
  </tr>
  <tr>
    <td>Commercial</td>
    <td>< 660 cc</td>
    <td>Yellow</td>
    <td>Black</td>
  </tr>
  <tr>
    <td>Commemorative</td>
    <td>–</td>
    <td>Green</td>
    <td>Multiple</td>
  </tr>
  <tr>
    <td>Glowing</td>
    <td>–</td>
    <td>Neon Green</td>
    <td>White</td>
  </tr>
</table>


## Datasets
### License Plates Dataset
* Dataset comprising 350 vehicles and their corresponding license plate bounding boxes for fine-tuning YOLOv8 detection model to detect license plates from images.

### alpr_jp
* Dataset comprising 1000+ Japanese license plate images for training character recognition models.
* Google Search images were used to supplement the dataset in case of missing or less common markings.


## Training
### Model
* CNN adapted from *Chinese License Plate Recognition System Based on Convolutional Neural Network*, layer depths adjusted according to specific recognition task.
### Hardware
* Apple M1 with MPS hardware acceleration

### Hyperparameters
* Number of epochs: 100
* Optimizer: Adam
* Initial learning rate: 1e-3
* Learning rate scheduler: StepLR, reduce by factor of 0.1 every 30 epochs
* Loss function: CrossEntropyLoss
* Random seed: 42

### Data Augmentation
 * Training images were passed to a 7-step augmenetation pipeline to enhance the model's robustness against image quality, camera angles and color variations.

![Augmentation pipeline](./images/HIMwhOP3XxY.png)

## Metrics
### License Plate Detection (YOLOv8)
<table>
  <tr>
    <th>Precision(B)</th>
    <th>Recall(B)</th>
    <th>mAP50(B)</th>
    <th>mAP50-95(B)</th>
    <th>Fitness</th>
  </tr>
  <tr>
    <td>0.84940</td>
    <td>0.75702</td>
    <td>0.85128</td>
    <td>0.63573</td>
    <td>0.65729</td>
  </tr>
</table>

### Character Recognition
<table>
  <tr>
    <th>Recognition Task</th>
    <th>Convolutional Layer Depths</th>
    <th>Samples (Classes)</th>
    <th>Accuracy</th>
    <th>Weighted F1</th>
    <th>Params (×10<sup>3</sup>)</th>
    <th></th>
  </tr>
  <tr>
    <td>① Region Name</td>
    <td style="white-space: nowrap;">32, 64, 128<br>32, 64, 128, 256<br>16, 32, 64, 128</td>
    <td>412 (134)<br><br><br></td>
    <td>0.93046<br>0.97816<br>0.97330</td>
    <td>0.92476<br>0.97543<br>0.97289</td>
    <td>368<br>462<br>132</td>
    <td><br>✅<br>(a)</td>
  </tr>
  <tr>
    <td>② Classification Number</td>
    <td style="white-space: nowrap;">32, 64, 128<br>16, 32, 64</td>
    <td>444 (11)<br><br></td>
    <td>0.97478<br>0.98423</td>
    <td>0.97760<br>0.98298</td>
    <td>97.9<br>25.9</td>
    <td>(b)<br>✅</td>
  </tr>
  <tr>
    <td>③ Kana Character</td>
    <td style="white-space: nowrap;">32, 64, 128<br>32, 64, 128, 256<br>16, 32, 64, 128</td>
    <td>430 (43)<br><br><br></td>
    <td>0.95814<br>0.97907<br>0.97674</td>
    <td>0.95581<br>0.97776<br>0.97519</td>
    <td>143<br>400<br>103</td>
    <td><br>✅<br>(a)</td>
  </tr>
  <tr>
    <td>④ Designation Number</td>
    <td style="white-space: nowrap;">32, 64, 128<br>32, 64, 128, 256<br>16, 32, 64</td>
    <td>547 (11)<br><br><br></td>
    <td>0.99086<br>0.99269<br>0.99086</td>
    <td>0.99092<br>0.99271<br>0.99086</td>
    <td>104<br>395<br>29.4</td>
    <td><br>(b)<br>✅</td>
  </tr>
</table>

## Observations
* (a) Model not deployed despite comparable metrics as a substantial decrease in confidence in the predicted classes was observed.

* (b) Increasing the number of parameters resulted in marginal improvement or degraded performance.

## References
**alpr_jp**
<br>
Big thanks to dyama san for sharing the alpr_jp dataset.
<br>
https://github.com/dyama/alpr_jp

**License Plates Dataset**
<br>
https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn

**YOLOv8**
<br>
https://github.com/ultralytics/ultralytics

**Chinese License Plate Recognition System Based on Convolutional Neural Network**
<br>
H. Chen, Y. Lin, and T. Zhao, 'Chinese License Plate Recognition System Based on Convolutional Neural Network', Highlights in Science, Engineering and Technology, vol. 34, pp. 95–102, 2023.
<br>
https://www.researchgate.net/publication/369470024

**ナンバープレートの見方 (How to Read a Number Plate)**
<br>
https://wwwtb.mlit.go.jp/tohoku/jg/jg-sub29_1.html

## Fun Fact
This repository was created on [Leap Day 2024](https://doodles.google/doodle/leap-day-2024/).
