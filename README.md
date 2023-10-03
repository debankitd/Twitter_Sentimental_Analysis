# Twitter-Tweets-Sentimental-Analysis-ML-Project

**This project is centred around analyzing user sentiments expressed on the Twitter social media platform using Machine Learning models, including Logistic Regression, Decision Tree, and Random Forest. The primary objective is to create a robust ML model capable of accurately predicting user sentiments (positive or negative) based on two key factors: user ratings and the textual content of their reviews.**
#

## Problem Aimed to Solve
**The project aims to serve as a valuable tool for social media platforms, providing them with actionable insights to shape their strategies, enhance user satisfaction, and expand their user base. To achieve this, the project follows a structured approach, beginning with data collection and preprocessing using Pandas, followed by ML model development with NLTK, RE, and Pandas. Model performance is then compared using metrics like Accuracy Score, and a comprehensive analysis of the dataset is conducted using Pandas and Seaborn to uncover hidden insights**


<br>
<br>
<p align="center"><a><img src="https://forthebadge.com/images/badges/built-with-love.svg"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a></p>

#  <img src="https://user-images.githubusercontent.com/106439762/181935629-b3c47bd3-77fb-4431-a11c-ff8ba0942b63.gif" width="48" height="48"> **User's Manual**

| Files| Description |
| ------------- | ------------- |
| **Twitter-Tweets-Sentimental-Analysis.ipynb** | This file contains the Python codes of the Suitable structuring of the Data, ML model creation and their Analysis parts. |
| **TTSAP - Data - Cleaning.ipynb** | This file contains the Python codes of the Data Cleaning and Exploratory Analysis parts. |
| **twitter_tweets.csv**  | This file provides the raw data for the project .  |
| **TTSAP_Final_Data.csv**  | This file contains the final data drawn out after the Data Cleaning.  |
<br>

# <img src="https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif" width="48" height="48" > Quick Start

    1. Started with cleaning the Data using Pandas by Dropping the Duplicates and Null values.
    
    2. Created a new Data Frame with only relevant columns out of the initial Data Frame.
 
    3. Also Replaced values in Ratings columns to Numbers based on their Intensity and exported the Cleaned data as CSV to use it further for Sentimental Analysis.
    
    4. Imported the Cleaned data in another ipynb file for Sentimental Analysis. 
    
    5. Performed Sentiment Intensity Analysis and based on it's score, categorized the Data as Positive or Negative and made a column out of it in the Data Frame and performed analysis on it.
    
    6. Removed special charcters and stop words present in the text and performed Vectorization (feature extraction) on it.
   
    7. Built different ML models like Logistic Regression, Decision Tree and Random Forest and compared them on the basis of their Accuracy Score and other Key Metrics, to find out the best fit model for the task.
   
<br>


# <img src="https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif" width="48" height="48" > Data Preperation

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/fb5420b1-e478-4e85-9668-d64b943d80c8)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/021713d9-eee8-4a01-aeb4-9ac858566af7)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/7aa57b57-112c-4327-8991-5da7810194ec)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/4be40a48-09c0-41c8-b341-41552230ccd9)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/d3b41a1f-034b-4edf-980f-493ed7ae3f49)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/68809d5c-c5fe-47a2-b6b5-9306dbcc0ea8)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/90a81813-4618-426e-8a28-ae60e9927705)

#  <img src=https://user-images.githubusercontent.com/106439762/178428775-03d67679-9aa4-4b08-91e9-6eb6ed8faf66.gif  width="48" height="48"> Analysis

o       Analysed the Accuracy Scores of Different ML Models for predicting the Sentiments.

o	Also analysed the F1 Scores, Precision Scores and Recall Scores of Different ML Models by creating their Classification Reports.

o	Performed comparison analysis to find out the Best Fit Machine Learning Model suitable for this task.

o	Analysed the relationship between the sentiments and ratings of the users.

o	Analysed the  Sentiments on the Cleaned Data 

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/841c8cb1-baa8-4d94-970f-d41ed73a9e41)


# <img src=https://user-images.githubusercontent.com/106439762/178428775-03d67679-9aa4-4b08-91e9-6eb6ed8faf66.gif  width="48" height="48"> ML Model Preparation

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/66ef4c11-ea0a-4994-8ade-99d63c5b21f6)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/ba2b6697-e873-45bc-ade5-69e45807950a)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/1cd9ffb9-b2c9-4394-a933-900d336d81c3)

![image](https://github.com/debankitd/Twitter_Sentimental_Analysis/assets/96534823/a513b8d8-bc37-4cd8-a181-80a26f651271)




 #  <img src=https://user-images.githubusercontent.com/106439762/178803205-47a08ce7-2187-4f96-b301-a2b68690619a.gif width="48" height="48" > Metrics
![pyhton-pandas](https://user-images.githubusercontent.com/106439762/177094844-d74edfa1-823d-4f17-8d94-3600e058cf1e.svg)
   
   1. The Accuracy Score of the Logistic Regression Model built for predicting Sentiments of the Users is found around 76% and the F1 Score is found 75%.
   
   2. The Accuracy Score of the Decision Tree Model built for predicting Sentiments of the Users is found around 73% and also the F1 Score is found 73%.
   
   3. The Accuracy Score of the Random Forest Model built for predicting Sentiments of the Users is found around 90% and also the F1 Score is found 90%.
   
   <br>


# <img src="https://user-images.githubusercontent.com/108053296/185756908-fbb62168-d923-48f2-992f-b8e2fde848fe.gif" width="48" height="48" > Conclusions
   
   1. Around 38% of the users have Negative Sentiments, whereas rest 62% have Positive Sentiments for Twitter's Services. This shows healthy User Satisfaction.
   
   2. Users providing Ratings as 1 are having highest Negative Sentiment bearing User Population.
   
   3. Whereas, those Users who provided Ratings as 3 are having highest Positive Sentiment bearing User Population.
   
   4. The Random Forest ML Model with about 90% Accuracy Score is found the Best Fit Model for this task.
   
   <br>
   
  
   
   
   #  <img src=https://user-images.githubusercontent.com/106439762/178803205-47a08ce7-2187-4f96-b301-a2b68690619a.gif width="48" height="48" > Future Scope
   
   <B> The aim of the Project is to: </B>
   
   1. Assist the Social media platforms with the Information that they can use to formulate their strategy in order to enhance the User's satisfaction.
   
   2. Advise the social media platforms on their relationships with their users and help them in expanding their user base.
   
   3. To provide the social media platforms a tool for keeping track on their User's sentiments.
   
   
   
    

![image](https://user-images.githubusercontent.com/108053296/189940016-b2f9ffd2-ff3c-46a7-90a0-ac2929953469.png)
