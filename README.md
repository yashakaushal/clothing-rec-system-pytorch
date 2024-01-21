# Clothing Product Recommendation using Deep Learning Models in PyTorch 

_Keywords_ - Neural Networks, EDA, data bias, Sparse AutoEncoders (SAC), Restricted Boltzmann Machine (RBM), Recommendation system

## Context 

Neural networks (NN) are powerful in predicting new customer preferences based on historic data. <br>
NN based recommendation systems have shown to outperform other traditional ML methods. <br>
In this notebook, we will use real world data of clothing sales and customers to make predictions for future customer choices. <br>
Along with predictions, we will perform Exploratory Data Analysis (EDA) to gain some valuable business insights. <br>

Step 1 - Exploratory Data Analysis (EDA) <br>
Step 2 - Reducing data to tackle class imbalance (CI) and difference in proportion of lables (DPL) <br>
Step 3 - Building Simple feedforward NN to predict future customer preferences <br>
Step 4 - Building Recommednation System using Sparse AutoEncoders (SAE) <br>
Step 5 - Building Recommednation System using Retricted Boltzman Machine (RBM) <br>

## Business Questions 

* Can we recommend similar clothing items to existing customers based on their age groups and past ratings?
* Can we predict which type of clothing a new customer would like and recommend them those to increase our sales?
* Are there specific clothing items or classes that receive consistently positive or negative reviews?
* What are the most popular clothing classes based on the number of reviews?

Age Group Preferences: <br>

* Are there certain age groups that are more likely to rate clothing items positively?
* Do certain age groups tend to give higher or lower ratings?
* Do different age groups prefer different types of clothing?
* Are there certain age groups that consistently recommend specific clothing items?
* Is there a correlation between the age of customers and the ratings they provide?

## Data 

* This data has been obtained from Kaggle - https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
* This dataset includes 23486 rows and 10 feature variables, out of which we will be using following 5 for our use case - 

1. _Clothing ID_: Integer Categorical variable that refers to the specific piece being reviewed.
2. _Age_: Positive Integer variable of the reviewers age.
3. _Rating_: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
4. _Recommended IND_: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
5. _Class Name_: Categorical name of the product class name.

## Methods 

This notebook makes extensive use of Deep Learning expertise learned from the [_Deep Learning A-Z_ Specialization](https://github.com/yashakaushal/my-certificates/blob/main/UC-92c361d8-f84d-49ac-b3c2-e65eb5fee8b1.pdf) <br>
Following methods are used - 
1. **Feedforward Neural Network** - Simple ANN with input + hidden + output layers, forward propagation, no cycle or loop of imformation flow. 
2. **Restricted Boltzman Machine** - Stochastic ANN with visible + hidden layers designed for unsupervised learning, ideal for feature learning and collaborative filtering 
3. **Autoencoders** - ANN designed for unsupervised learning consisting of an encoder and a decoder with the goal to learn a compressed, efficient representation of input data
