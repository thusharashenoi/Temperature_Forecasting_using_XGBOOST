# Temperature_Forecasting_using_XGBOOST
## Project Description:
Applying cutting-edge deep learning and machine learning techniques for Temperature forecasting has significantly improved Temperature prediction over conventional approaches. These novel techniques are appropriate for handling sizable data sets in forecasting scenarios where sizable amounts of historical temperature datasets could be used.  The project that follows is an Intel Optimised XGBOOST based Time series forecasting model that focuses on the prediction of Temperatures based on the information readily accessible from 1750 to 2015. It mainly focusses on demonstrating the impact of Global Warming on the Global Average Temperature of the earth, causing slight deviation from the regular temperature patterns followed in the past. 
## Table of Contents:
  1. Data used
  2. Prerequisites
  3. Introduction to XGBoost
      - Introduction
      - Benefits
      - Syntaxes
      - Installation
  4. Intel API AI Analytics Toolkit
      - Componenets
      - Benefits
  5. Process of Model Building
## Data: 
Geological records show that there have been a number of large variations in the Earth's climate. These have been caused by many natural factors, including changes in the sun, emissions from volcanoes, variations in Earth's orbit and levels of carbon dioxide (CO2). 

The dataset had the following columns:
1. date
2. LandAverageTemp
3. LandAverageTempUncertainity
4. LandMaxTemp
5. LandMaxTempUncertainity
6. LandMinTemp
7. LandMinTempUncertainity
8. Land&OceanAverageTemp
9. Land&OceanAverageTempUncertainity

The following columns will be used in order to develop the model:
1. date
2. LandAverageTemp


##  Prerequisites:
 This model uses the following libraries of Python as Prerequisites:
 1. Numpy
      - Python library used for working with arrays.
      - syntax to use library: `import numpy as np`
 2. Pandas
      - Python library used for working with data sets.
      - syntax to use library: `import pandas as pd`
 3. Seaborn
      - Python library for making statistical graphics.
      - syntax to use library: `import seaborn as sns`
 4. Matplotlib
      - It is a cross-platform, data visualization and graphical plotting library for Python and its numerical extension NumPy.
      - syntax to use library `import matplotlib.pyplot as plt`
 5. Skelearn
      - Scikit-learn (Sklearn) is the most useful and robust library for ML which provides a selection of efficient tools for ML and statistical modeling         including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.
      - syntax to use library: `from sklearn.metrics import mean_squared_error, mean_absolute_error`
 6. XGBoost
      - It is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library.
      - syntax to use library: `import xgboost as xgb`
    
  ## Introduction to XGBoost:
  XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It         provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems. XGBoost first grasps     the machine learning concepts and algorithms that XGBoost builds upon: supervised machine learning, decision trees, ensemble learning, and gradient       boosting.Using XGBoost on Intel CPUs takes advantage of software accelerations powered by oneAPI, without requiring any code changes. Software            optimizations deliver the maximum performance for your existing hardware. This enables faster iterations during development and training, and lower     latency during inference.
   - BENEFITS:
   1. XGBoost is a highly portable library on OS X, Windows, and Linux platforms. It's also used in production by organizations across various verticals,         including finance and retail.
   2. XGBoost is open source, so it's free to use, and it has a large and growing community of data scientists actively contributing to its development.         The library was built from the ground up to be efficient, flexible, and portable.
   - SYNTAXES: 
      The following are the syntaxes for creating XGB Classifiers and XGB Regressors respectively:
   
   `xgb_cl = xgb.XGBClassifier()`
   
   `reg = xgb.XGBRegressor(n_estimators=1000)`
  - INSTALLATION:
  Intel-optimized XGBoost can be installed in the following ways:
  As a part of Intel® AI Analytics Toolkit
  From PyPI repository, using pip package manager: pip install xgboost
  From Anaconda package manager:
    -  Using Intel channel: conda install xgboost –c intel
    -  Using conda-forge channel: conda install xgboost –c conda-forge
  As a Docker container (provided you have a DockerHub account)

  ## Components and uses of Intel API AI Analytics Toolkit.
 - Components:
  
      - Machine Learning
      
          - Intel Extension for Scikit-Learn
          
          - Intel Optimised XGBoost
          
      -  Intel Optimized Python
      
          - Numpy
          
          - Scipy
          
          - Numba
          
          - Pandas
          
          - Data Parallel Python
      - Data Analytics
     
          - Intel Distribution of Modin
          
          - Omni Sci Backend
      - Deep Learning
          
          - Intel Extension for TensorFlow
          
          -Intel Extension for PyTorch
          
          - Model Zoo for Intel Architecture
          
          - Intel Neural Compressor
- Benefits:
    - High performance pipeline for Machine Learning and Deep Learning.
    - Fast, accurate and efficient training and deployment of your model.
    - The ability to scale up and out for distributed computing.
    - Interoperability with intel’s latest optimization in a single package.
   
## Components and uses of Intel API AI Analytics Toolkit
1. Finalising a project topic

  - The first step into starting this project was to decide from the variety of domains which topic we wanted to work on. We were asked to choose from:       Predicting disease outcomes, medical image analysis, telemedicine, image recignition, fraud detection, recommendation system, time series analysis,       customer segmentation.
  - We decided to go ahead with Time Series Analysis for Global Temperature Prediction. 
   
2. Choosing an appropriate toolkit

  - From the the various above listed Intel API toolkits, we decided to use the AI Analytics Toolkit in order to use the Intel Optimized XGBoost for           building predictive models.
  
3. Finding a data set

  - After going through multiple datasets on Kaggle, we finaz=lid on the Global Mean Temperatures.csv dataset.
  
4. Cleaning the data set

  - Used Forward fill and Backward fill Methods in order to eliminate the null values to improve accuracy of the dataset. 
 
5. Data visualisation

  - Use of Scatter plots to demontrate the trends of Globabl Average temperature variation with time.
  
6. Splitting data set for training and testing data
  - The dataset was then segregated into two part - Training data and Testing Data.
  - The split was made on the date 01/Jan/1969
   
7. Visualisation of training and testing data
  - The Dataset is again Visualized using Scatter plots.
  - Now the Training and Testing datasets are shown separately using blue(testing data) and orange(training data)
  
8. Creation of XGBoost model

9. Training the model

10. Testing the model

11. Error calculation
    -Use of mean error and mean absolute error in order to show the accuracy of the model.
    
12. Documentation
