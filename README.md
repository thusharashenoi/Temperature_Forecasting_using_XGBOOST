# Temperature_Forecasting_using_XGBOOST
## Project Description:
Applying cutting-edge deep learning and machine learning techniques to weather forecasting has significantly improved weather prediction over conventional approaches. These novel techniques are appropriate for handling sizable data sets in forecasting scenarios where sizable amounts of historical weather datasets could be used.  The project that follows is an Intel Optimised XGBOOST based Time series forecasting model that focuses on the prediction of Temperatures until__________ based on the information readily accessible from 1750 to 2015.
## Table of Contents:
  1. Prerequisites
  2. A brief Introduction to XGBoost
  3. Components and Benefits of Intel API AI Analytics Toolkit.
  4. Brief Description of the process
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
  XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It         provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems. XGBoost first grasps     the machine learning concepts and algorithms that XGBoost builds upon: supervised machine learning, decision trees, ensemble learning, and gradient         boosting.
   - BENEFITS:
   1. XGBoost is a highly portable library on OS X, Windows, and Linux platforms. It's also used in production by organizations across various verticals,         including finance and retail.
   2. XGBoost is open source, so it's free to use, and it has a large and growing community of data scientists actively contributing to its development.         The library was built from the ground up to be efficient, flexible, and portable.
   - SYNTAXES: 
      The following are the syntaxes for creating XGB Classifiers and XGB Regressors respectively:
   
   `xgb_cl = xgb.XGBClassifier()`
   
   `reg = xgb.XGBRegressor(n_estimators=1000)`

  ## Components and uses of Intel API AI Analytics Toolkit.
  -Components:
  
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
##  Process of Model Building:
