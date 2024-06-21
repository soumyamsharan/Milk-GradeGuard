
# Milk Grade-Guard

Enhancing food safety through machine learning




## Table of Contents
- [About The Project](#about-the-project)
  - [Tech Stack](#tech-stack)
  - [File Structure](#file-structure)
- [Literature Review](#literature-review)
  - [Milk grading parameters](#milk-grading-parameters)
  - [Machine Learning](#machine-learning)
  - [Classification Model](#classification-model)
  - [KNN](#knn)
  - [Naive Bayes](#naive-bayes)
  - [SVM](#svm)
  - [Gradient Boosting](#gradient-boosting)
- [Problem Statement](#problem-statement)
- [Implementation](#implementation)
  - [Methodology](#methodology)
  - [Result Analysis](#result-analysis)
- [Future Scope](#future-scope)
- [Author](#author)
- [References](#references)

## About The Project
Milk Grade-Guard provides a multifaceted strategy to improve food safety. The solution can eliminate human error, save costs related to manual testing, and streamline operations by automating quality assessments using machine learning. 
More significantly, less product waste is produced and dangerous milk is kept
from reaching customers thanks to early problem detection enabled by ML- powered prediction. In order to confirm Milk Grade-Guard's function in strengthening food safety protocols within the dairy industry, more investigation and development are necessary
  - ### Tech Stack
    - **Python:** A versatile programming language used for general-purpose programming.
    - **Jupyter Notebook:** An interactive web application for creating and sharing computational documents.
  - ### File Structure
    - Milk-GradeGuard


## Literature Review

- ### Milk grading parameters: 
  Our machine learning approach for assessing milk quality, Milk Grade-Guard, depends on a number of factors to reliably       predict milk grade. Below is a summary of the primary parameters that will be employed:

  ***Fat Content:*** One of the most important markers of milk quality is its fat content. Deviations from the anticipated 
  range could indicate problems with the health of the cows or water dilution.<br/>***pH Level:*** The pH level of fresh 
  milk is slightly acidic. Significant departures from this range may be a sign of faulty storage or spoiling brought on by 
  bacterial development.<br/>***Temperature:*** During storage and transit, milk should be kept within a certain range. 
  Deviations may damage quality by hastening the growth of microorganisms.<br/>***Color:*** Depending on the breed and 
  nutrition, natural milk color can change slightly. Severe discolouration may indicate contamination or spoiling. 
  <br/>***Taste:*** While taste tests may not be feasible for a large-scale system, the machine learning model may benefit 
  from past data on flavor assessments made by human specialists. Odd tastes may be a sign of infection or spoiling. 
  <br/>***Odor:*** Like taste, odor data (provided by human specialists) can be used to train models. Odd smells can 
  indicate the presence of microorganisms or spoiling.<br/>***Turbidity:*** Clarity-impairing contaminants might alter the 
  quality of milk. Turbidity readings are used by Milk Grade-Guard into its machine learning model to help it forecast milk 
  grade more accurately.
