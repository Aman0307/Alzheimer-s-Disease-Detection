# Alzheimer-s-Disease-Detection Using MRI Scans

This repository contains a Jupyter Notebook that focuses on the detection of Alzheimer's Disease using machine learning techniques. Alzheimer's Disease is a progressive brain disorder that affects memory, thinking skills, and behavior. Early detection and diagnosis are crucial for managing the disease effectively.

## Notebook Overview
The Jupyter Notebook provided in this repository aims to detect Alzheimer's Disease using machine learning algorithms. The notebook is structured as follows:

## Introduction: 

This research aims to enhance the early detection and diagnosis of Alzheimer's disease (AD) through the utilization of a machine learning-based approach that focuses on analyzing the hippocampus region in MRI scans. Traditional diagnostic methods for AD often rely on subjective evaluations, while MRI scans provide objective and quantitative data on brain structure. As the hippocampus, a key region associated with memory and learning, undergoes significant changes in individuals with AD, it presents valuable insights for disease detection. By extracting relevant features from the hippocampus region in MRI scans, these features can serve as reliable indicators of AD progression. Machine learning algorithms can be employed to develop a predictive model that enables non-invasive and cost-effective methods for AD diagnosis, ultimately leading to improved patient outcomes, advancements in AD research, and more effective treatment strategies. accurate AD detection. The outcome of this research has the potential to contribute to the development of AD

## Data: 

The ADNI 3 (Alzheimer's Disease Neuroimaging Initiative) dataset is a highly valuable and widely utilized resource in the field of Alzheimer's disease research. It is a longitudinal multicenter study conducted in North America with the primary objective of identifying and tracking biomarkers associated with the progression of Alzheimer's disease. The dataset comprises a comprehensive collection of clinical, neuroimaging, and genetic data from individuals classified into different diagnostic categories, including Alzheimer's disease (AD), healthy controls (CN), and mild cognitive impairment (MCI).
                                                                               
For the purpose of this research, a subset of the ADNI 3 dataset was employed, consisting of 73 images from individuals diagnosed with AD, 75 images from healthy control subjects (CN), and 71 images from individuals with mild cognitive impairment (MCI). These images are most likely MRI scans of the brain, which serve as essential tools in Alzheimer's disease research for identifying structural and  anatomical changes associated with disease progression. By utilizing this carefully selected subset of the ADNI 3 dataset, the research enables the development and evaluation of machine learning algorithms or other computational techniques for the early detection and diagnosis of Alzheimer's disease based on neuroimaging data. The balanced distribution of images across the AD, CN, and MCI classes ensures that the model can effectively learn and distinguish between the different diagnostic categories.


## Summary: 

This research project focuses on the development of a machine learning model for the classification of brain MRI images. The goal is to accurately distinguish between patients with Alzheimer's disease (AD), individuals with normal cognitive function (CN), and those with mild cognitive impairment (MCI). The study utilizes a dataset consisting of MRI images obtained from the AD, CN, and MCI categories. The research workflow involves several steps. First, the MRI images are preprocessed to extract patches from different planes (axial, coronal, and sagittal) and apply intensity thresholding. The patches are then saved as numpy files for further analysis. Next, a portion of the dataset is set aside for testing purposes, while the remaining samples are used for training the machine learning model.  The classification model is based on a convolutional neural network architecture. It consists of multiple convolutional layers followed by batch normalization and max-pooling operations. The extracted features are flattened and passed through a series of dense layers with batch normalization. Finally, the output layer with softmax activation performs the three-class classification.

The model is trained using the training dataset, and the validation dataset is utilized to monitor the model's performance and prevent overfitting. The training process is optimized using the Adam optimizer and a sparse categorical cross-entropy loss function. Early stopping is employed as a regularization technique to halt training if the model's performance on the validation set stagnates. The results of this research project demonstrate the effectiveness of the proposed machine learning model in classifying brain MRI images into the AD, CN, and MCI categories. The model achieves a high accuracy rate, indicating its potential as a diagnostic tool for the early detection and differentiation of neurodegenerative diseases. Further studies can be conducted to evaluate the model's performance on larger datasets and explore its applicability in clinical settings.

Overall, this research contributes to the field of medical imaging analysis and provides a foundation for the development of intelligent systems for automated diagnosis and prognosis of cognitive disorders based on brain MRI 



## To run the notebook locally, follow the steps below:

Clone the repository to your local machine using the following command:


Copy code  <br />
```git clone https://github.com/Aman0307/Alzheimer-s-Disease-Detection.git```  <br />
 <br />
Ensure that you have Jupyter Notebook installed. If not, install it using the following command:

Copy code  <br />
```pip install jupyter```  <br />
Open a terminal or command prompt and navigate to the cloned repository directory.  <br />

Launch Jupyter Notebook by entering the following command:

Copy code  <br />
```jupyter notebook```  <br />
A browser window will open, displaying the contents of the repository. Click on the Jupyter Notebook file (Alzheimer's_Disease_Detection.ipynb) to open it.  <br />

Follow the instructions in the notebook to execute each cell and proceed with the analysis.

## Dependencies
The notebook relies on the following libraries and frameworks:

* Python (3.7 or above)
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

Ensure that these dependencies are installed before running the notebook.


## Contributing
Contributions to this repository are welcome. If you have suggestions for improvements, bug fixes, or new features, please submit a pull request. Additionally, if you encounter any issues or have questions, feel free to open an issue.

## License
This project is licensed under the MIT License.
