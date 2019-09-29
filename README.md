## Objective
1) Credit Card Spend Analysis
2) Finding Categories User is interested in [ Used - recency, type of transaction]
3) Find where and when User can spend next?
4) Supervised model to predict if a cardholder is doing to make a purchase in the
next k days

**Input** -
Transaction Data

**Output** :
1) Data is Analysed
2) Model is saved in pkl format

**Note**: Python code is pep8 compliant

## Tools use 
> Python 3

> Main Libraries Used -
1) numpy
2) pandas
3) scikit

## Installing and Running

> 
```sh
$ cd customer_transaction_prediction
$ pip install -r requirements.txt
``` 

For Running Script
```sh
$ python model.py
```

## Various steps in approach are -

Major steps required for modeling is ​-
1. Data wrangling
2. Data transformation for supervised model for user level feature set (Feature Engineering). 
    Features are -
    1. Recency
    2. Number of Transactions before
    3. Total Amount of transactions before 
    4. Difference between transaction days mean 
    5. Difference between transaction days variance
3. Building machine learning model and evaluation. It includes
    a. Scaling - It is used to bring all features at the same scale.
    b. Find Best Machine learning model
    c. Hyperparamter tuning using grid search
    d. Confusion matrix

Various Techniques implied for preprocessing -
1. Remove unused columns for decreasing data storage
2. Rename Columns to standardised name for easy use
3. Taken care of edge cases while input for handling error
4. String change to upper case so as to group unique values
5. Date parameter into datetime format for easy computation
6. Remove rows having amount in negative as it is wrong input. Can take absolute
value but it depends from where data come
7. Remove numerical values in vandor, Cardholder_First_Initial and
Cardholder_Last_Name as some of the values are added as a suffix or prefix due to
some problem so after cleaning we can group them easily
8. Check proportion of null values
9. Remove rows having null values in columns so that analysis can be done taking that
features
10. Join Cardholder_First_Initial and Cardholder_Last_Name as user so as to find users.

Segmented users according to where they spend and how much so that we can target them accordingly. We can make user category depending on user spending in many ways -
1. Users which spend on various options
2. Users spend on one thing but in high amount
3. Users which do high number of transaction
4. User spend on one thing but frequently

Major categories which defined are -
1. Basic Users: Customers who do less transaction and less frequently with low amount 
2. Mid Users: Amount is not high, mid frequent
3. Important Users: High Amount, very frequent and high transactions.

Other Major categories in which defined are -
1. New users - Find source from where they come and then target 
2. Existing users - Find churn prediction and then action