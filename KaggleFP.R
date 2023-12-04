library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(ggmosaic)
library(tidymodels)
library(embed)
library(discrim)
library(naivebayes)
library(kernlab)
library(themis)
library(doParallel)
library(keras)
library(tensorflow)

rm(list=ls())
## read in the data
fp_train <- vroom("./train.csv") %>%
  select(-c(id)) %>%
  mutate(target = as.factor(target))
fp_test <- vroom("./test.csv")


## check for missing values
sum(is.na(fp_train))


#################
## Naive Bayes ##
#################

my_recipe <- recipe(target ~., data = fp_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  #step_normalize(all_predictors()) %>%
  #step_smote(all_outcomes(), neighbors = 50) %>%
  prep()

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid_nb <- grid_regular(Laplace(),
                               smoothness())

folds <- vfold_cv(fp_train, v = 10, repeats = 1)

nb_results_tune <- nb_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_nb,
            metrics=metric_set(accuracy))


bestTune <- nb_results_tune %>%
  select_best("accuracy")


final_wf_nb <- nb_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=fp_train)

nb_predictions <- final_wf_nb %>%
  predict(new_data = fp_test, type = "prob")


fp_predictions_nb <- nb_predictions %>%
  bind_cols(., fp_test) %>%
  select(c(id, .pred_Class_1, .pred_Class_2, .pred_Class_3, .pred_Class_4, .pred_Class_5, .pred_Class_6, .pred_Class_7, .pred_Class_8, .pred_Class_9)) %>%
  rename(Class_1 = .pred_Class_1) %>%
  rename(Class_2 = .pred_Class_2) %>%
  rename(Class_3 = .pred_Class_3) %>%
  rename(Class_4 = .pred_Class_4) %>%
  rename(Class_5 = .pred_Class_5) %>%
  rename(Class_6 = .pred_Class_6) %>%
  rename(Class_7 = .pred_Class_7) %>%
  rename(Class_8 = .pred_Class_8) %>%
  rename(Class_9 = .pred_Class_9)


vroom_write(x=fp_predictions_nb, file="./nb.csv", delim=",")
############## 
## 20.73874 ##
##############
####################
## Random Forests ##
####################

cl <- makeCluster(8)

registerDoParallel(8)

my_recipe <- recipe(target ~., data = fp_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  #step_smote(all_outcomes(), neighbors = 50) %>%
  prep()

my_mod_crf <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

class_reg_tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_crf)

tuning_grid_crf <- grid_regular(min_n(),
                                mtry(range = c(1, 10)))


folds <- vfold_cv(fp_train, v = 10, repeats = 1)

CV_results_crf <- class_reg_tree_wf %>%
  tune_grid(resamples = folds, 
            grid = tuning_grid_crf,
            metrics=metric_set(accuracy))

bestTune <- CV_results_crf %>%
  select_best("accuracy")

final_wf_crf <- class_reg_tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=fp_train)

rf_predictions <- predict(final_wf_crf, new_data = fp_test, type = "prob")


fp_predictions_rf <- rf_predictions %>%
  bind_cols(., fp_test) %>%
  select(c(id, .pred_Class_1, .pred_Class_2, .pred_Class_3, .pred_Class_4, .pred_Class_5, .pred_Class_6, .pred_Class_7, .pred_Class_8, .pred_Class_9)) %>%
  rename(Class_1 = .pred_Class_1) %>%
    rename(Class_2 = .pred_Class_2) %>%
    rename(Class_3 = .pred_Class_3) %>%
    rename(Class_4 = .pred_Class_4) %>%
    rename(Class_5 = .pred_Class_5) %>%
    rename(Class_6 = .pred_Class_6) %>%
    rename(Class_7 = .pred_Class_7) %>%
    rename(Class_8 = .pred_Class_8) %>%
    rename(Class_9 = .pred_Class_9)



vroom_write(x=fp_predictions_rf, file="./rf.csv", delim=",")


stopCluster(cl)
######################
## 500 trees .55304 ##
######################
#############################
## Support Vector Machines ##
#############################

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

tuning_grid_svm <- grid_regular(cost(),
                                rbf_sigma(),
                                levels = 5)


folds <- vfold_cv(fp_train, v = 5, repeats = 1)


svm_results_tune <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_svm,
            metrics=metric_set(accuracy))


bestTune <- svm_results_tune %>%
  select_best("accuracy")


final_wf_svm <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=fp_train)

svm_predictions <- final_wf_svm %>%
  predict(new_data = fp_test, type = "class")


fp_predictions_svm <- svm_predictions %>%
  bind_cols(., fp_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)



test <- fp_predictions_svm %>%
  mutate(Class_1 = if_else(type == "Class_1", 1, 0)) %>%
  mutate(Class_2 = if_else(type == "Class_2", 1, 0)) %>%
  mutate(Class_3 = if_else(type == "Class_3", 1, 0)) %>%
  mutate(Class_4 = if_else(type == "Class_4", 1, 0)) %>%
  mutate(Class_5 = if_else(type == "Class_5", 1, 0)) %>%
  mutate(Class_6 = if_else(type == "Class_6", 1, 0)) %>%
  mutate(Class_7 = if_else(type == "Class_7", 1, 0)) %>%
  mutate(Class_8 = if_else(type == "Class_8", 1, 0)) %>%
  mutate(Class_9 = if_else(type == "Class_9", 1, 0)) %>%
  select(-c(type))


vroom_write(x=test, file="./svm.csv", delim=",")


#########################
## K Nearest Neighbors ##
#########################

my_recipe <- recipe(target ~., data = fp_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  #step_smote(all_outcomes(), neighbors = 50) %>%
  prep()


knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')


knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_mod)


tune_grid <- grid_regular(neighbors(),
                          levels = 5)

folds <- vfold_cv(fp_train, v = 4, repeats = 1)

CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tune_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")


final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=fp_train)

knn_predictions <- final_wf %>%
  predict(new_data = fp_test, type = "prob")


fp_predictions_knn <- knn_predictions %>%
  bind_cols(., fp_test) %>%
  select(c(id, .pred_Class_1, .pred_Class_2, .pred_Class_3, .pred_Class_4, .pred_Class_5, .pred_Class_6, .pred_Class_7, .pred_Class_8, .pred_Class_9)) %>%
  rename(Class_1 = .pred_Class_1) %>%
  rename(Class_2 = .pred_Class_2) %>%
  rename(Class_3 = .pred_Class_3) %>%
  rename(Class_4 = .pred_Class_4) %>%
  rename(Class_5 = .pred_Class_5) %>%
  rename(Class_6 = .pred_Class_6) %>%
  rename(Class_7 = .pred_Class_7) %>%
  rename(Class_8 = .pred_Class_8) %>%
  rename(Class_9 = .pred_Class_9)


vroom_write(x=fp_predictions_knn, file="./knn.csv", delim=",")


#############
## 1.51913 ##
#############

#################
## Neural Nets ##
#################

nn_recipe <- recipe(target ~., data = fp_train) %>%
  step_mutate(target = as.factor(target), skip = TRUE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1) %>%
  prep()

nn_model <- mlp(hidden_units = tune(),
                epochs = 300) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tune_grid <- grid_regular(hidden_units(range = c(1, 10)))

folds <- vfold_cv(fp_train, v = 5, repeats = 1)

nn_results_tune <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tune_grid,
            metrics = metric_set(accuracy))

#nn_results_tune %>% collect_metrics() %>%
#  filter(.metric=="accuracy") %>%
#  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune_nn <- nn_results_tune %>%
  select_best("accuracy")

final_nn_wf <- nn_wf %>%
  finalize_workflow(bestTune_nn) %>%
  fit(data = fp_train)

nn_predictions <- final_nn_wf %>%
  predict(new_data = fp_test, type = "class")

fp_predictions_nn <- nn_predictions %>%
  bind_cols(., fp_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)



test <- fp_predictions_nn %>%
  mutate(Class_1 = if_else(type == "Class_1", 1, 0)) %>%
  mutate(Class_2 = if_else(type == "Class_2", 1, 0)) %>%
  mutate(Class_3 = if_else(type == "Class_3", 1, 0)) %>%
  mutate(Class_4 = if_else(type == "Class_4", 1, 0)) %>%
  mutate(Class_5 = if_else(type == "Class_5", 1, 0)) %>%
  mutate(Class_6 = if_else(type == "Class_6", 1, 0)) %>%
  mutate(Class_7 = if_else(type == "Class_7", 1, 0)) %>%
  mutate(Class_8 = if_else(type == "Class_8", 1, 0)) %>%
  mutate(Class_9 = if_else(type == "Class_9", 1, 0)) %>%
  select(-c(type))


vroom_write(x=test, file="./nn.csv", delim=",")

######################
## 200 epoch 8.1009 ##
######################
##############
## Boosting ##
##############

my_recipe <- recipe(target ~., data = fp_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  #step_smote(all_outcomes(), neighbors = 50) %>%
  prep()


library(bonsai)
library(lightgbm)
library(dbarts)

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boosting_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

boost_tune_grid <- grid_regular(tree_depth(),
                                trees(),
                                learn_rate(),
                                levels = 3)

folds <- vfold_cv(fp_train, v = 5, repeats = 1)

boost_results_tune <- boosting_wf %>%
  tune_grid(resamples = folds,
            grid = boost_tune_grid,
            metrics = metric_set(accuracy))

bestTune_boost <- boost_results_tune %>%
  select_best("accuracy")

final_boost_wf <- boosting_wf %>%
  finalize_workflow(bestTune_boost) %>%
  fit(data = fp_train)

boost_predictions <- final_boost_wf %>%
  predict(new_data = fp_test, type = "prob")

fp_predictions_boost <- boost_predictions %>%
  bind_cols(., fp_test) %>%
  select(c(id, .pred_Class_1, .pred_Class_2, .pred_Class_3, .pred_Class_4, .pred_Class_5, .pred_Class_6, .pred_Class_7, .pred_Class_8, .pred_Class_9)) %>%
  rename(Class_1 = .pred_Class_1) %>%
  rename(Class_2 = .pred_Class_2) %>%
  rename(Class_3 = .pred_Class_3) %>%
  rename(Class_4 = .pred_Class_4) %>%
  rename(Class_5 = .pred_Class_5) %>%
  rename(Class_6 = .pred_Class_6) %>%
  rename(Class_7 = .pred_Class_7) %>%
  rename(Class_8 = .pred_Class_8) %>%
  rename(Class_9 = .pred_Class_9)


vroom_write(x=fp_predictions_boost, file="./boost.csv", delim=",")

###########################
## predict class 5.82863 ##
## predict prob  0.54054 ##
###########################

## :)


