SVM
================

``` r
train <- readRDS("./data/cog_train_preproc.RDS")
test <- readRDS("./data/cog_test_preproc.RDS")
```

First I look at an SVM with a lienar kernel.

``` r
set.seed(1)
linear.tune <- e1071::tune.svm(cdr~., 
                               data = train, 
                               kernel = "linear", 
                               cost = exp(seq(-5, 1, len = 20)))
# summary(linear.tune)

best.linear <- linear.tune$best.model
summary(best.linear)
```

    ## 
    ## Call:
    ## best.svm(x = cdr ~ ., data = train, cost = exp(seq(-5, 1, len = 20)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  0.1584834 
    ##       gamma:  0.1111111 
    ## 
    ## Number of Support Vectors:  370
    ## 
    ##  ( 187 183 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  NonDementia Dementia

``` r
train.pred <- predict(best.linear, newdata = train)

train.confusion = confusionMatrix(data = train.pred, 
                                  reference = train$cdr)

pred.linear <- predict(best.linear, newdata = test)

test.confusion = confusionMatrix(data = pred.linear, 
                                 reference = test$cdr)
```

The training accuracy is 0.7729 (0.7392, 0.8043), with a Kappa or 0.389. The test accuracy is 0.762 (0.7125, 0.8068), with a Kappa of 0.369.

Next, I will look at an SVM with a radial kernel to see whether it is a better fit.

``` r
set.seed(1)
radial.tune <- e1071::tune.svm(cdr~., 
                               data = train, 
                               kernel = "radial", 
                               cost = exp(seq(-4, 5, len = 10)),
                               gamma = exp(seq(-8, -3, len = 5)))

# summary(radial.tune)


best.radial <- radial.tune$best.model
summary(best.radial)
```

    ## 
    ## Call:
    ## best.svm(x = cdr ~ ., data = train, gamma = exp(seq(-8, -3, len = 5)), 
    ##     cost = exp(seq(-4, 5, len = 10)), kernel = "radial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  148.4132 
    ##       gamma:  0.00117088 
    ## 
    ## Number of Support Vectors:  364
    ## 
    ##  ( 185 179 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  NonDementia Dementia

``` r
train.pred.radial <- predict(best.radial, newdata = train)

train.confusion = confusionMatrix(data = train.pred.radial, 
                                 reference = train$cdr)

pred.radial <- predict(best.radial, newdata = test)

test.confusion = confusionMatrix(data = pred.radial, 
                                 reference = test$cdr)
```

The training accuracy is 0.7865 (0.7533, 0.817), with a Kappa of 0.4317. The test accuracy is 0.762 (0.7125, 0.8068), which is the same as with the linear kernel. The Kappa value is 0.381.

Overall, it looks like the predictive capability on test data is similar between models. The difference lies in the training accuracy, which is slightly higher in the SVM with the radial kernel. Since we're interested in accuracy, I would choose the SVM with radial kernel.
