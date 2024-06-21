
# Milk Grade-Guard

Enhancing food safety through machine learning




## Table of Contents
- [About The Project](#about-the-project)
  - [Tech Stack](#tech-stack)
  - [File Structure](#file-structure)
- [Basic Concepts](#basic-concepts)
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


## Basic Concepts

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

- ### Machine Learning
  Machine learning is  a quickly emerging field of technology which enables computers to automatically learn from the          historical data. Machine learning uses various algorithms to create mathematical models and forecasts based on information 
  or historical data. Nowadays it is being used for many different things, like image recognition, recommender systems, 
  email filtering, and speech recognition.
  Machine learning can be broadly classified into three types. These classification is based on the nature of the learning 
  system and the data available. The types are as follows:
    - Supervised learning:Labeled data is used to teach models how to predict outcomes.
    - Unsupervised Learning: In this process, algorithms sift through unlabeled data in search of patterns or clusters.
    - Reinforcement Learning: In this approach, models gain decision-making skills by acting in a way that maximizes a 
      concept of cumulative reward.

- ### Classification Model
  The supervised ML are classified into 2 types namely Regression model and Classification model . The classification 
  algorithms in machine learning are essential for categorizing data into predefined classes. Before using the model to make 
  predictions on newly discovered data, it must first be thoroughly trained on training data and assessed on test data. Some 
  popular classification algorithms include Logistic regression , Naive bayes classification , KNN , Decision trees, SVM .

- ### KNN
  The KNN algorithm is commonly employed as a classification algorithm, based on the premise that comparable points can be 
  located next to one another, while it can be used for regression or classification problems as well.<br/><br/>

  The distance between a query point and the other data points must be computed in order to ascertain which data points are 
  closest to a particular query point. The decision boundaries that divide query points into various areas are formed in 
  part by the below distance measurement formula.

  Distance Euclidean (p=2): This distance metric is restricted to real-valued vectors and is the most widely used one. It 
  measures a straight line between the query location and the other point being measured using the formula below.<br/>
  <img src = "https://almablog-media.s3.ap-south-1.amazonaws.com/image_12_71a43363e2.png" width = "200"/>

- ### Naive Bayes
  Naive Bayes Classifier algorithm calculates conditional probabilities. This algorithm is based on Bayes’ Theorem with the 
  assumption of independence among input variables. It is known for high efficiency in handling large datasets. It is used 
  widely in applications like spam filtering, text classification, and recommendation systems.<br/>
  The formula used for calculation in Naive Bayes classifier is as follows:<br/>
  P(C|X) = P(X|C)P(C) / P(X)<br/>
  as the above formula assumes the input variables to be independent. We consider, P(X|C)=P(X1|C) P(X2|C)...P(Xn|C)<br/>
  ( P(C|X) ) is the posterior probability of class ( C ) given predictor ( X ).<br/>
  ( P(X|C) ) is the likelihood which is the probability of predictor ( X ) given class ( C ).<br/>
  ( P(C) ) is the prior probability of class.<br/>
  ( P(X) ) is the prior probability of predictor.<br/>

- ### SVM
  Support Vector Machine (SVM) is a supervised learning algorithm used for classification tasks. It is used split the data 
  into 2 or more classes by creating a hyperplane. It works by finding the optimal hyperplane that maximizes the margin 
  between different classes. In two-dimensional space, this hyperplane is a line which divides the points into two 
  categories. In higher dimensions, it’s a plane. SVM can be linear and non-linear. For creating the decision boundary, 
  we can different SVM kernels like linear, polynomial, RBF etc.<br/>
  The  SVM creates a hyperplane defined by the equation:<br/>
  w.x+b=0 <br/>
  Where the <br/>
  ( w ) is the weight vector. <br/>
  ( x ) represents the input features. <br/>
  ( b ) is the bias term. <br/> The main aim is to maximize the margin to reduce the noise and error between the classes. 
  This can be done by, max(2/||w||) subject to y(w.+b) >= 1.

- ### Gradient Boosting
  Gradient Boosting is a machine learning technique used for both classification tasks. It builds an ensemble of decision 
  trees, in a sequential manner where each tree tries to correct the errors made by the previous one. The method involves 
  training trees on the residual errors of the predecessor, effectively refining the model with each iteration. Thus is 
  very efficient in terms of classification models. <br/>
  F2(x) = F1(x) +n. H1(x)  <br/>where <br/>
  F2(x) is the updated model after the 1st iteration. <br/>
  F1(x) is model at the 1st iteration. <br/>
  n is the learning rate <br/>
  H1(x) is the weak learner fitted on the gradient of the  loss function at 1st iteration. <br/>
  
