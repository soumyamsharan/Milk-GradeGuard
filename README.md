
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

## Problem Statement
For the sake of consumer confidence and public health, milk product quality and safety must always be guaranteed. However, human examination and subjective assessment are frequently used in conventional techniques of evaluating milk quality, which leaves opportunity for error and inconsistent results. In addition, the complexity of the supply chain and the rising demand for dairy products make it difficult to uphold strict quality control standards. 
<br/>
The problem statement in this instance centers on the requirement for a dependable and efficient system to improve food safety in the dairy sector through the application of machine learning (ML) technologies. Creating a Milk Grade-Guard system that can reliably and effectively evaluate milk quality at every stage of production and distribution is the main problem.

## Implementation
  - ### Methodology
    <img src = "https://github.com/soumyamsharan/Milk-GradeGuard/blob/main/Diagrams/WhatsApp%20Image%202024-04-07%20at%2020.34.03_6851c303.jpg" width = "500"/><br/>
    - **Data Collection**<br/>
        - Source: <br/>
A organized milk quality dataset from Kaggle was used. The dataset ought to have multiple factors that impact the quality of milk, including pH, temperature, taste, odor, fat content, turbidity, and color.

        - Data Preprocessing: <br/>i.Duplicate Removal: To avoid skewing the training process for our machine learning 
          models, we had remove duplicate entries from the dataset. <br/>
          ii. Missing Value Handling: If there are few missing values, we use appropriate methods such as mean/median 
          imputation or deletion to deal with the missing data points (null values). <br/>
          iii. Outlier Identification and Management: We locate and address outliers, or extreme values, that may cause the 
          underlying trends in the data to become distorted. <br/>
          iv. Scaling: The dataset's features may be measured in several units. Normalization and standardization are 
          scaling strategies that guarantee every feature contributes equally to the training process of the model.<br/>
    - **Model Training, Evaluation and Selection** <br/>
        - Model Training: <br/>
        Our dataset was partitioned into 70:30 training and testing sets. The model is trained on the training set and 
        evaluated on unseen data using the testing set. To prepare for classification problems, we investigate and variety 
        of machine learning algorithms.The correlations between the milk quality characteristics and the appropriate 
        quality grades (such as "high","medium”,“low”) found in the labeled data will be discovered using these algorithms.
        - Evaluation of the Models: <br/>
        To evaluate our model, we examined four algorithms: Gradient Boosting, Support Vector Machines (SVM), Naive Bayes, 
        and K-Nearest Neighbors (KNN). We determined which technique produced the best results .Accuracy, precision, 
        recall, and F1-score are just a few of the metrics that will be used to thoroughly assess each trained model's 
        performance. Algorithms of the classification models used are as follow: <br/>

            1) ***SVM***<br/>
                ```markdown
                IMPORT libraries<br/>

                LOAD dataset<br/>
                SELECT features and target<br/>

                SPLIT dataset into training and testing sets<br/>

                INITIALIZE SVM with RBF kernel<br/>
                SET parameters gamma and C<br/>
                FIT SVM on training data<br/>

                PREDICT using SVM on test data<br/>

                CALCULATE accuracy and F1 score<br/>
                PRINT performance metrics <br/>
                ```
            2) ***NAIVE BAYES***<br/>
                ```markdown
                IMPORT libraries<br/>
                SUPPRESS warnings<br/>
    
                STANDARDIZE features<br/>
    
                SPLIT dataset into training and testing sets<br/>
                PRINT shapes of the splits<br/>
    
                INITIALIZE Gaussian Naive Bayes classifier<br/>
                FIT classifier on training data<br/>

                PREDICT on test data<br/>
                CALCULATE accuracy<br/>
    
                PRINT accuracy and classification report<br/>

                ```
            3) ***KNN***<br/>
                ```markdown
                IMPORT libraries<br/>
                
                LOAD dataset<br/>
                SELECT features and target<br/>
                
                SPLIT dataset into training and testing sets<br/>
                
                INITIALIZE KNN classifier with specified number of neighbors<br/>
                FIT KNN on training data<br/>
                
                PREDICT using KNN on test data<br/>

                CALCULATE confusion matrix and classification report<br/>
                PRINT evaluation metrics<br/>
                ```
            4) ***GRADIENT BOOSTING***<br/> 
                ```markdown
                IMPORT libraries<br/> 
                
                LOAD dataset<br/> 
                SELECT features and target<br/> 
                
                SPLIT dataset into training and testing sets<br/> 
                PRINT sizes of the splits<br/> 
                
                INITIALIZE Gradient Boosting Classifier<br/> 
                FIT classifier on training data<br/> 
                
                PREDICT on test data<br/> 
                
                PRINT classification report<br/>
                ```
        - Model Selection: <br/>
        We choose the Gradient Boosting model as it performs best in terms of accuracy and dependability when it comes to 
        determining milk quality based on the evaluation findings. The Milk Grade-Guard system's output will be based on 
        this model. The accuracy of all the models are as follows: <br/>
        ```markdown
        |     Algorithms  	    |    Accuracy Percentage   	|
        |---------------------------|---------------------------|
        |        KNN        	    |           68%           	|
        |        SVM        	    |          55.56%         	|
        |    Naive Bayes    	    |           72%           	|
        | Gradient Boosting 	    |           88%           	|
        ```
    - **Testing** <br/>
      The Milk Grade-Guard system's output will be based on this model. To evaluate the chosen model's practicality, we 
      test it on the input dataset from user.
  - ### Result Analysis
    <img src = "https://github.com/soumyamsharan/Milk-GradeGuard/blob/main/Diagrams/op.png" width = "500"/><br/>
    This shows the comparison of the accuracy among the models used.
    This graph clearly concludes that Gradient Boosting(0.88) has the highest accuracy among all. Thus, this is the best 
    fit model. <br/><br/>
    <img src = "https://github.com/soumyamsharan/Milk-GradeGuard/blob/main/Diagrams/input.png" width = "500"/><br/>
    This shows the prediction of quality of milk by taking the set of input from users. Here the model is being tested over 
    a particular set of data.<br/><br/>
    <img src = "https://github.com/soumyamsharan/Milk-GradeGuard/blob/main/Diagrams/graph.png" width = "500"/> <br/>
    This Graph is used for illustrating the relationship between the actual values (y_test) and the predicted values (y_pred).<br/><br/>
    
## Future Scope
As we all know , food safety is a very critical aspect of public health . It is very necessary for protecting consumers from hazards like microbiological contamination, chemicals and other toxins.
Through this project, we contributed to the betterment of public health by creating an algorithm that can test the given sample of milk and catagorize it into good, bad, average milk. 
This project has a very promising future scope. It can potentially revolutionize in the dairy sector and ensure high standard milk are served to consumers.<br/>
The future developments to this project can be as follows:<br/><br/>
a)Expansion to other dairy products: By Extending this algorithm, we can help to monitor and improve the quality of other dairy products like cheese , cottage cheese(paneer), yogurt etc.

b) Real-time analysis: by implementing the real-time monitoring system using this algorithm can help to provide us with instant feedback and monitoring of dairy farms.

c) Consumer apps: By linking the project and extending it with apps, so as to help the consumer to verify the milk they are consuming is healthy or not.

d) Advanced analysis: By integrating with much more sophisticate machine learning algorithms, we can also predict the number of days the current "good" quality milk will turn into "bad" milk.

These are some of the future advancements that can be done to enhance the "Milk grade-guard" project.
This will surely help to become an integral part of dairy industry's push towards innovation and sustainability.

## Author
[Soumyam Sharan](https://github.com/soumyamsharan)

## References
[1] Habsari.W , Udin.F and Arkeman.Y “An analysis and design of fresh
    milk smart grading system based on internet of things.” IOP Conference Series: Earth and Environmental Science. Vol. 
    1063. No. 1. IOP Publishing, 2022.<br/><br/>
[2] S. Kumari, M. K. Gourisaria, H. Das and D. Banik, “Deep Learning Based Approach for Milk Quality Prediction,” 2023 11th 
    International Conference on Emerging Trends in Engineering and Technology - Signal and Information Processing (ICETET - 
    SIP), Nagpur, India, 2023, pp. 1-6, doi: 10.1109/ICETET-SIP58143.2023.10151626. <br/><br/>
[3] Bentejac, Candice , Cs ´ org ¨ o, Anna and Mart ˝ ´ınez-Munoz, Gonzalo. “A ˜ Comparative Analysis of XGBoost”.(2019)<br/><br/>
[4] Septiani, Winnie, and Tatit K. Bunasor. “Intelligent System for Pasteurized Milk Quality Assessment and Prediction.” 
    (2007).<br/><br/>
[5] Ganesh Kumar, P., A. Alagammai, B. S. Madhumitha, and B. Ishwariya. ”IoT based milk monitoring system for the detection 
    of milk adulteration.” ESP J Eng Technol Adv 2, no. 2 (2022): 6-9. <br/><br/>
[6] Guo, Gongde, Hui Wang, David Bell, Yaxin Bi, and Kieran Greer. “KNN model-based approach in classification.” In On The 
    Move to Meaningful Internet Systems 2003: CoopIS, DOA, and ODBASE: OTM Confederated International Conferences, CoopIS, 
    DOA, and ODBASE 2003, Catania, Sicily, Italy, November 3-7, 2003. Proceedings, pp. 986-996. Springer Berlin Heidelberg, 
    2003. <br/><br/>
[7] Vimalajeewa, Dixon, Chamil Kulatunga, and Donagh P. Berry. “Learning in the compressed data domain: Application to milk 
    quality prediction.” Information Sciences 459 (2018): 149-167.<br/><br/>
[8] Mahesh, Batta. “Machine learning algorithms-a review.” International Journal of Science and Research (IJSR).[Internet] 
    9, no. 1 (2020): 381-386.<br/><br/>
[9] Cunha, Matheus Henrique Lopes, and Hygor Santiago Lara. “Machine learning as a way to have more accuracy when defining 
    milk quality classification.” Caderno de ANAIS HOME (2023).<br/><br/>
[10] Bhavsar, Drashti, Yash Jobanputra, Nirmal Keshari Swain, and Debabrata Swain. “Milk Quality Prediction Using Machine 
     Learning.” EAI Endorsed Transactions on Internet of Things 10 (2024).<br/><br/>
[11] Rish, Irina. “An empirical study of the naive Bayes classifier.” In IJCAI 2001 workshop on empirical methods in 
     artificial intelligence, vol. 3, no. 22, pp. 41-46. 2001.<br/><br/>
[12] Yang, Yongheng, and Lijuan Wei. “Application of E-nose technology combined with artificial neural network to predict 
     total bacterial count in milk.” Journal of Dairy Science 104, no. 10 (2021): 10558-10565.<br/><br/>
[13] Voskoboynikova, Olga, Aleksey Sukhanov, and Axel Duerkop. “Optical pH sensing in milk: A small puzzle of indicator 
     concentrations and the best detection method.” Chemosensors 9, no. 7 (2021): 177.<br/><br/>
[14] Kumar, JNVR Swarup, D. N. V. S. L. S. Indira, Kalyanapu Srinivas, and MN Satish Kumar. “Quality assessment and grading 
     of milk using sensors and neural networks.” In 2022 International Conference on Electronics and Renewable Systems 
     (ICEARS), pp. 1772-1776. IEEE, 2022.<br/><br/>
[15] Saravanan, S., M. Kavinkumar, N. S. Kokul, N. S. Krishna, and V. I. Nitheeshkumar. “Smart milk quality analysis and 
     grading using IoT.” In 2021 5th International Conference on Intelligent Computing and Control Systems (ICICCS), pp. 
     378-83. IEEE, 2021.<br/><br/>
[16] Vishwanathan, S. V. M., and M. Narasimha Murty. “SSVM: a simple SVM algorithm.” In Proceedings of the 2002 
     International Joint Conference on Neural Networks. IJCNN’02 (Cat. No. 02CH37290), vol. 3, pp. 2393-2398. IEEE, 2002.<br/><br/>
[17] Toko, Kiyoshi. “Research and development of taste sensors as a novel analytical tool.” Proceedings of the Japan 
     Academy, Series B 99, no. 6 (2023): 173-189.<br/><br/>
