Analysis Plan
================
Laura Cosgrove
5/1/2019

# Analysis Plan

We aim to diagnose Dementia/Non-Dementia using count of APOE
risk/protective alleles and anatomical volumes derived from Freesurfer
analysis of MRI scans, adjusting for height, weight, and age.

*I don’t think we should use MMSE (which is another way dementia is
assessed) in our models*. We also shouldn’t use `year_round` or
`subject`. However, they are still currently in the cog\_data file.

We will compare a series of models:

  - Logistic regression

  - Linear discriminant analysis

  - K-nearest neighbors

  - Naive Bayes

  - Classification and regression trees (CART)

  - Gradient boosting machine

  - Random forest classifier

  - Support vector machines
