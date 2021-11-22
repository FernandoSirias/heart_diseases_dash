# Heart Diseases

**Data sorce:** [UCI Heart Disease Data, Kaggle](https://www.kaggle.com/redwankarimsony/heart-disease-data)

This project uses the UCI dataset to create a dash app that can ***predict*** based on the information provided if the person is at risk of heart disease or not.

You can see the data cleaning, EDA, and model creation in the **notebook**.

We deploy the application to the web using heroku. To see live the dash app click this [link](https://heartdiseasesdash.herokuapp.com/).

## Data description

This is a multivariate type of dataset which means providing or involving a variety of separate mathematical or statistical variables, multivariate numerical data analysis. It is composed of 14 attributes which are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, oldpeak — ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels and Thalassemia. This database includes 76 attributes, but all published studies relate to the use of a subset of 14 of them. The Cleveland database is the only one used by ML researchers to date. One of the major tasks on this dataset is to predict based on the given attributes of a patient that whether that particular person has heart disease or not and other is the experimental task to diagnose and find out various insights from this dataset which could help in understanding the problem more.

## Features description

- **id:** Unique id for each patient.
- **age:** Age of the patient in years.
- **origin:** place of study.
- **sex:** Male/Female.
- **cp:** chest pain type [typical angina, atypical angina, non-anginal, asymptomatic].
- **trestbps:** resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital)).
- **chol:** serum cholesterol in mg/dl.
- **fbs:** if fasting blood sugar > 120 mg/dl.
- **restecg:** resting electrocardiographic results -- Values: [normal, stt abnormality, lv hypertrophy].
- **thalach:** maximum heart rate achieved.
- **exang:** exercise-induced angina (True/ False).
- **oldpeak:** ST depression induced by exercise relative to rest.
- **slope:** the slope of the peak exercise ST segment.
- **ca:** number of major vessels (0-3) colored by fluoroscopy.
- **thal:** [normal; fixed defect; reversible defect].
- **num:** the predicted attribute.
