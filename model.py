import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle


# do not show warnings
import warnings
warnings.filterwarnings("ignore")


def main():
    # list file in folder
    path = os.getcwd()
    files = os.listdir(path + '/data')
    csv_file = [f for f in files if f[-3:] == 'csv']

    # importing data
    data_df = pd.read_csv(path + '/data/' + csv_file[0], skiprows=0, sep=',',
                          keep_default_na=False,
                          na_values=["", "n/a", "na", "--", "NaN"])

    if data_df.shape[0] == 0:
        return

    # rename column for analysis
    data_df.columns = data_df.columns.str.replace(' ', '_')
    data_df.columns = data_df.columns.str.replace('-', '_')

    # drop unused columns
    data_df = data_df.drop(['Year_Month'], 1)

    # change to upper case so as to group same names
    cols_to_upper = [
        'Agency_Name',
        'Cardholder_Last_Name',
        'Cardholder_First_Initial',
        'Description',
        'Vendor',
        'Merchant_Category_Code_(MCC)']
    for col in cols_to_upper:
        data_df[col] = map(
            lambda x: x.upper(),
            data_df[col])

    # conveted to date time
    data_df['Transaction_Date'] = pd.to_datetime(data_df['Transaction_Date'])
    data_df['Posted_Date'] = pd.to_datetime(data_df['Posted_Date'])

    # removing amount having negative values
    data_df = data_df[data_df.Amount >= 0]

    # replace number with empty string
    data_df['Vendor'] = data_df['Vendor'].str.replace(r'\d+', '')
    data_df['Cardholder_Last_Name'] = data_df['Cardholder_Last_Name'].str.replace(
        r'\d+', '')
    data_df['Cardholder_First_Initial'] = data_df['Cardholder_First_Initial'].str.replace(
        r'\d+', '')

    # replace empty string with null
    data_df = data_df.replace(r'^\s*$', np.nan, regex=True)

    data_df['User'] = data_df['Cardholder_First_Initial'] + \
        ' ' + data_df['Cardholder_Last_Name']

    # find number of null values and its percent
    print data_df.isnull().sum()
    print data_df.isnull().mean().round(4) * 100

    # drop rows having null values in any column
    data_df = data_df.dropna(how='any', axis=0)

    # histogram
    data_df['Amount'].astype(int).plot(kind='hist')
    plt.show()

    # Top users based on their monthly spending
    print data_df.groupby([data_df['Transaction_Date'].dt.month, 'User'])['Amount'].sum().reset_index(
        name='Amount').sort_values(['Transaction_Date', 'Amount'], ascending=False).groupby('Transaction_Date').head(5)

    # Top most popular vendors by the number of transactions per quarter
    data_df_temp = data_df.copy()
    data_df_temp['count'] = 1
    print data_df_temp.groupby([data_df_temp['Transaction_Date'].dt.quarter, 'Vendor'])['count'].sum().reset_index(
        name='count').sort_values(['Transaction_Date', 'count'], ascending=False).groupby('Transaction_Date').head(5)

    # Top highest categories by the sum of amounts
    print data_df.groupby(
        ['Merchant_Category_Code_(MCC)'])['Amount'].sum().reset_index(
        name='Amount').sort_values(
            ['Amount'],
        ascending=False).head(5)

    # split data in two dataset for making different features
    data_before = data_df[(data_df.Transaction_Date <
                           date(2014, 5, 1))].reset_index(drop=True)
    data_next = data_df[(data_df.Transaction_Date >=
                         date(2014, 5, 1))].reset_index(drop=True)

    # finding numbe of unique users
    data_user = pd.DataFrame(data_before['User'].unique())
    data_user.columns = ['User']

    # create dataframe with user and first transaction date in data_next
    data_next_first_purchase = data_next.groupby(
        'User').Transaction_Date.min().reset_index()
    data_next_first_purchase.columns = ['User', 'FirstPurchaseDate']

    # create dataframe with user and last transaction date in data_before
    data_before_last_purchase = data_before.groupby(
        'User').Transaction_Date.max().reset_index()
    data_before_last_purchase.columns = ['User', 'LastPurchaseDate']

    # find recency in before data in days
    data_before_last_purchase['Recency'] = (
        data_before_last_purchase['LastPurchaseDate'].max() -
        data_before_last_purchase['LastPurchaseDate']).dt.days
    data_user = pd.merge(data_user, data_before_last_purchase[[
                         'User', 'Recency', 'LastPurchaseDate']], on='User', how='left')

    # merge two dataframes
    purchase_dates = pd.merge(
        data_before_last_purchase,
        data_next_first_purchase,
        on='User',
        how='left')

    # calculate difference in days between two purchase
    purchase_dates['NextTransDate'] = (
        purchase_dates['FirstPurchaseDate'] -
        purchase_dates['LastPurchaseDate']).dt.days
    data_user = pd.merge(
        data_user, purchase_dates[['User', 'NextTransDate']], on='User', how='left')

    # get number of transaction for each user
    trans_frequency = data_before.groupby(
        'User').Transaction_Date.count().reset_index()
    trans_frequency.columns = ['User', 'Frequency']

    # add frequency column to user
    data_user = pd.merge(data_user, trans_frequency, on='User', how='left')

    # get total amount for each user
    total_amount = data_before.groupby('User').Amount.sum().reset_index()
    total_amount.columns = ['User', 'Amount']

    # add amount column to user
    data_user = pd.merge(data_user, total_amount, on='User', how='left')

    # fill with large values who have null values so as to use for model
    data_user = data_user.fillna(1111)

    # create dataframe with User and Date
    data_before_day = data_before[['User', 'Transaction_Date']]
    # convert Transaction Datetime to day
    data_before_day['TransactionDay'] = data_before['Transaction_Date'].dt.date
    data_before_day = data_before_day.sort_values(['User', 'TransactionDay'])
    # drop duplicates as can be two transaction same day
    data_before_day = data_before_day.drop_duplicates(
        subset=['User', 'TransactionDay'], keep='first')

    # last two transactions
    data_before_day['DiffDay'] = (
        data_before_day['TransactionDay'] -
        data_before_day.groupby('User')['TransactionDay'].shift(1)).dt.days

    # mean and standard deviation of difference between transactions in days
    user_day_diff = data_before_day.groupby('User').agg(
        {'DiffDay': ['mean', 'std']}).reset_index()
    user_day_diff.columns = ['User', 'DiffDayMean', 'DiffDayStd']
    data_user = pd.merge(data_user, user_day_diff, on='User', how='left')

    # drop rows having null in mean and variance
    data_user = data_user.dropna()

    # float to int
    cols = ['NextTransDate']
    data_user[cols] = data_user[cols].applymap(np.int64)

    # drop unused columns
    data_user = data_user.drop(['LastPurchaseDate', 'User'], axis=1)
    input_days = [1, 5, 10]
    for k in input_days:
        # making category of user depending on number of days for next
        # transaction
        print 'For Days = ' + str(k)
        data_user_temp = data_user.copy()
        data_user_temp['NextTransDateCategory'] = 0
        data_user_temp.loc[data_user_temp.NextTransDate <=
                           k, 'NextTransDateCategory'] = 1
        data_user_temp = data_user_temp.drop(['NextTransDate'], axis=1)

        # separate class for modeling
        X = data_user_temp.drop(['NextTransDateCategory'], axis=1)
        y = data_user_temp.NextTransDateCategory

        # split data as train and test data
        train_set, test_set, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=25)

        # Transform features between 0 and 1
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_set)
        X_train = scaler.transform(train_set)
        X_test = scaler.transform(test_set)

        # create an array of models
        models = []
        models.append(("LR", LogisticRegression()))
        models.append(("NB", GaussianNB()))
        models.append(("RF", RandomForestClassifier()))
        models.append(("Dtree", DecisionTreeClassifier()))

        # measure the accuracy using cross validation and to check if any
        # outlier or not
        for name, model in models:
            kfold = KFold(n_splits=2, random_state=22)
            cv_result = cross_val_score(
                model, X_train, y_train, cv=kfold, scoring="accuracy")
            print(name, cv_result)

        # gridsearch to find best parameters
        search_params = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'n_estimators': [100, 200, 300, 1000]
        }
        gsc = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=search_params,
            cv=5,
            error_score='numeric')
        gsc.fit(X_train, y_train)

        # model parameters
        best_params = gsc.best_params_
        print "Best Params: ", gsc.best_params_
        # log likelihood score for grid search
        print "Best Log Likelihood Score: ", gsc.best_score_

        # using grid search best parameter for final model
        final_model = RandomForestClassifier(
            max_depth=best_params["max_depth"],
            n_estimators=best_params["n_estimators"],
            random_state=False,
            verbose=False)
        final_model.fit(X_train, y_train)
        print('Accuracy of Random Forest Classifier on training set: {:.2f}'
              .format(final_model.score(X_train, y_train)))
        print('Accuracy of Random Forest Classifier on test set: {:.2f}'
              .format(final_model.score(X_test, y_test)))

        # finding prediction on test data and calculate confusion matrix
        predictions = final_model.predict(X_test)
        conf_mat = confusion_matrix(y_test, predictions)
        print(conf_mat)

        # Exporting Model
        with open(str(k) + '_day_supervised_model.pkl', 'wb') as fid:
            pickle.dump(final_model, fid)


if __name__ == "__main__":
    main()
else:
    print ("Executed when imported")
