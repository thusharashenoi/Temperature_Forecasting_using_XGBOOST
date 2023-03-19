# Temperature_Forecasting_using_XGBOOST
## Project Description:
Applying cutting-edge deep learning and machine learning techniques for Temperature forecasting has significantly improved Temperature prediction over conventional approaches. These novel techniques are appropriate for handling sizable data sets in forecasting scenarios where sizable amounts of historical temperature datasets could be used.  The project that follows is an Intel Optimised XGBOOST based Time series forecasting model that focuses on the prediction of Temperatures based on the information readily accessible from 1750 to 2015.
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

