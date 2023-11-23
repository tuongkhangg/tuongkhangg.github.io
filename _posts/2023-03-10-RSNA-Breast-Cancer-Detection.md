---
title:  "RSNA Screening Mammography Breast Cancer Detection"
date: 2023-03-10
permalink: /posts/2023-03-10-RSNA-Cancer-Detection/
tags: 
    - kaggle
    - cv
---


***Goal of the Competition***
The goal of this competition is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

Your work improving the automation of detection in screening mammography may enable radiologists to be more accurate and efficient, improving the quality and safety of patient care. It could also help reduce costs and unnecessary medical procedures.

# Table of contents
1. [Overview](#overview)   
    .....Context  
2. [Data Analysis](#analysis)  
    .....Overview  
    .....Key observation  
    .....Key observations about METADATA  
    .....View Features  
    .....Age Distribution  
    .....Label distribution  
3. [Methods](#method)  
    .....Auxiliary Network Model(#auxiliary)  
    .....Auxiliary Multi-view Model(#multiview)    
    .....Single View Model(#single)

# 1.Overview: <a name="overview"></a>

## Context
According to the WHO, breast cancer is the most commonly occurring cancer worldwide. In 2020 alone, there were 2.3 million new breast cancer diagnoses and 685,000 deaths. Yet breast cancer mortality in high-income countries has dropped by 40% since the 1980s when health authorities implemented regular mammography screening in age groups considered at risk. Early detection and treatment are critical to reducing cancer fatalities, and your machine learning skills could help streamline the process radiologists use to evaluate screening mammograms.

Currently, early detection of breast cancer requires the expertise of highly-trained human observers, making screening mammography programs expensive to conduct. A looming shortage of radiologists in several countries will likely worsen this problem. Mammography screening also leads to a high incidence of false positive results. This can result in unnecessary anxiety, inconvenient follow-up care, extra imaging tests, and sometimes a need for tissue sampling (often a needle biopsy).

The competition host, the Radiological Society of North America (RSNA) is a non-profit organization that represents 31 radiologic subspecialties from 145 countries around the world. RSNA promotes excellence in patient care and health care delivery through education, research, and technological innovation.

Your efforts in this competition could help extend the benefits of early detection to a broader population. Greater access could further reduce breast cancer mortality worldwide.

"kaggle: RSNA Screening Mammography Breast Cancer Detection"

# 2. Data Analysis: <a name='analysis'></a>

The dataset has 2 main parts:
* Image files
* Metadata
    
**Overview:**
* There are 54,706 training samples to work with, each containing 14 features.
    * There are only 2 different sites where imaging took place.
    * Of the 54,706 training samples, there are 11,913 unique patients, meaning that patients are represented multiple times.
    * There are 10 different machines that performed imaging.
* There are 4 testing samples to work with, each containing 9 features.

**Key observations**

* The testing set only contains 4 samples. 
* The testing set contains a "prediction_id" column that is not present in the training set.
* There are 5 features in the training set that don't appear in the testing set. There may be an opportunity here to use soft labeling to predict those features that don't appear in the testing set to help us with the classification task, if those features end up being at least partially correlated with the target variable. In particular, we may want to see if we can build classifiers that filter for:
    * biopsy
    * invasive
    * BIRADS
    * density
    * difficult_negative_case - this in particular may be a good starting point for generating a coarse first-pass filter
* Patients are represented multiple times


**Key observations about METADATA**

* Both "CompressionForce" and "BodyPartThickness" measurements may be useful for future investigation. While only observed in half of the image data we have, they still may add information that we can use.
* The "ContentDate" field may have provided us unique insights into when various imaging series have been undertaken. For example, we may have seen the progression of a disease or been able to determine whether a finding was difficult to ascertain based solely upon the date the image was taken. However, the field appears to have been sanitized.
* The number of "BitsStored" varies, meaning we're likely to have different qualities of images depending on the dicom file.
We have a wide variety of Rows and Columns, meaning that image sizes and resolutions are going to be varied. We'll probably need to normalize / standardize the resolution of the images. Need to be wary of generating compression artifacts or of losing information if we do so.
* Two different "PhotometricInterpretation" types are available. Again, we need to be aware this may result in different interpretations when we view the images themselves.
* The "VOILUTFunction" types are different. Briefly, VOI is a Value Of Interest, and LUT is a Look Up Table. VOI LUT functions specify how pixel intensity values should be presented when viewed. The lookup table values may have a non-linear relationship to the intensity value physically encoded in the image. Again, this can impact our perception of the image as we view it. We need to make sure we properly represent the dicom image when we display it or use it in a machine learning process.

**View Features**

<!--<img src="../assets/images/rsna-22/view_features.png" alt="View Feature">-->

We can see from the graph that the images are primarily comprised of two different views:
* **MLO** - Mediolateral Oblique View - captures the most breast tissue. The pectoral muscle is included in the view, and is used as a guide to assess proper patient positioning and overall image quality. The MLO view is taken looking downwards, but angled to look from the center of the chest outwards.
* **CC** -  Craniocaudal View - as with the Mediolateral Oblique View, the pectoral muscle may be included in the view, which again is used as a guide to assess proper patient positioning. The main difference however, is that the Crainocaudal View is taken from above the breast looking straight downwards (i.e. no angle is introduced as in the oblique view).

Both MLO and CC views are known as standard views. These views are the ones most commonly used in routine screening. It is worthwhile to note however, there may be contraindications where these views are not performed, such as when disease processes are present. With patients under 40, only the MLO of the left and right sides may be performed to reduce overall radiation exposure, since it adequately captures the most breast tissue.

<!--<img src="../assets/images/rsna-22/mlo_views.png" >-->

In addition to the standard views, there are other significantly less utilized views in the training set, which are separate and distinct from MLO and CC:

* **AT** - Unknown - possibly Tangent View?
<img src="../../images/rsna_breast/view_at.png" >
* **ML** - Mediolateral View - taken from the center of the chest between the breasts, looking outwards. Usually used in instances where the Mediolateral Oblique View has not been taken or is not possible. This is a favored view when the oblique view is not available, as most disease processes occur on the lateral side of the breast, and therefore will be closer to the film allowing for a clearer image of the pathology.
<img src="../../images/rsna_breast/view_ml.png" >
* **LM** - Lateromedial View - similar to the Mediolateral View, except the view is taken from the arms pointing inwards towards the chest. This view is not as ideal, again due to the tendency for pathology to occur on the lateral side of the breast.
<img src="../../images/rsna_breast/view_lm.png" >
* **MLO** - Lateromedial Oblique View - similar to MLO except taken from the outside of the body pointing inwards.
<img src="../../images/rsna_breast/view_mlo.png" >  

**Age Distribution**

The patient's age in years. The average age is 58 years old, with the vast majority of the patients having between 50 and 65 years old. There are a few outliers with very young patients (26-30 years old), as well as a few more senior patients (89 years old).

<img src="../../images/rsna_breast/age_distribution.png" >

Furthermore, tagehe youngest patient to have cancer is 38 years old, while the mean of those patients is 63 years old.

<img src="../../images/rsna_breast/age_distribution_02.png" >

**Label distribution**
<!--<img src="../assets/images/rsna-22/label_distribution.png" >-->

The classes are highly unbalanced which the labeled data is 1158 images.

# 3. Methods <a name='method'></a>
Some solutions was mentioned with distinct approaches in this competition:

* First method - we can combine metadata and images data to an auxiliary network with:
    * Metadata network could be a neural network with input is metadata
    * Images network (backbone could be Resnet, EfficientNet, EfficinetNet v2, RexNext, ...)
    * Concatenate both output to predict.

* Second method - using Multiview-model with metadata to compare between left and right view of image.
* Finally, I approached with the basic model based on ROI crop dataset.

So, I demonstrate each method, explain the idea, drawback and why it works.

### Auxilinary Network Model <a name='auxiliary'></a>
<img src="../../images/rsna_breast/auxliary_loss.png">

This strategy aims to focus on integrating the combined .CSV and .PNG data distributions in order to improve the main classifier's overall performance.  

1. Image Head: A specific type of model made to process photos.  
2. Auxiliary Head: This includes a simple linear model designed specifically to manage metadata.  

### Auxiliary Multi-view Model <a name='multiview'></a>

<img src='../../images/rsna_breast/aux_multi_view.png'>

Specifically, the MultiLateralityDualView model is made to examine and contrast the left and right breasts in order to detect malignancy. Our observations showed that different machine IDs were linked to significantly diverse visual styles. But with the information related to a certain patient, the visual style is always the same. Moreover, an interesting trend was observed: the majority of individuals only had cancer on one side of their breast. Taking these realizations into account, we have deliberately given the model the ability to evaluate and contrast the left and right sides, improving its capacity to more accurately forecast the existence of cancer.  

Unfortunately, this idea came to us in the later rounds of the competition, which meant we didn't have enough time to fully optimize and fine-tune the model. We used the complete dataset to train a model in spite of this time restriction, and we turned it in by the deadline. However, we continue to have faith in the MultiLateralityDualView model's potential and see it as important for future research and development directions.

### Single View Model <a name='single'></a>

<img src='../../images/rsna_breast/single_model.png'>

* **ROI Cropping**: 
The ROI cropping technique was used with great care to greatly improve the texture and detail retention at a set resolution. Using the YOLOX-nano 416x416 for ROI detection, this method has a clear benefit. When a deep learning (DL) detector is used instead of rule-based techniques, the result is bounding boxes with a more consistent aspect ratio and a smaller size. The DL detector is particularly good at narrowing its focus to the breast area, which helps to produce a more precise and accurate detection result.  
* **Augmentation**  
* **Upsampling**:  In all of my experiments, I upsample affirmative cases in every epoch.  
    
    Stabilizing training depends on ensuring that a batch or iteration has at least one positive case. When I set the number of affirmative cases per batch to 0.5, I've seen difficulties with training. Changing the upsampling ratio affects the prediction distribution and cross-validation scores differently for different backbones and hyperparameter selections. In spite of this, I would rather have the lowest positive-to-negative ratio that still ensures at least one positive case every atch and roughly matches the actual data distribution.  
      
    In the early epochs, a higher positive-to-negative ratio helps with faster training. To solve prediction distribution and threshold difficulties, I have experimented with linearly varying the positive-to-negative ratio across epochs, especially with EffB4. Nevertheless, I saw no change in the cross-validation findings in spite of these attempts.

*If you have any new research approach, I'd love to hear from you.*

