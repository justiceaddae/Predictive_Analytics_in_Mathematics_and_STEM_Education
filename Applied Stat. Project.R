# =============================================================================
# Applied Statistical Modeling and Predictive Analytics in Mathematics Education
#
# Dataset: UCI Student Performance (Mathematics course)
#          Cortez, P. & Silva, A. (2008), UCI Machine Learning Repository
#          395 students, two Portuguese secondary schools
#
# This script reproduces the regression, Cross Validation, classification, 
# feature-importance, and DESCRIPTIVE (GROUP-LEVEL ANALYSIS)
# results reported.
# =============================================================================

# ---- 0. Packages -----------------------------------------------------------
required_packages <- c(
  "randomForest", "caret", "pROC", "xgboost",
  "dplyr", "ggplot2", "vip", "purrr"
)

to_install <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

library(randomForest)
library(caret)
library(pROC)
library(xgboost)
library(dplyr)
library(ggplot2)
library(vip)
library(purrr)


set.seed(238)  # matches random_state=42 used in the Python analysis

# ---- 1. Load data ------------------------------------------------------------

dataload<-read.csv("D:\\USA\\UAF\\NIW\\New NIW\\R\\student-mat.csv")
str(dataload)


# ---- 2. Feature sets ---------------------------------------------------------
# "full"  = includes G1 and G2 (in-progress prediction)
# "early" = excludes G1 and G2 (early-warning prediction, before any grades exist)

target <- "G3"
all_features   <- setdiff(names(dataload), "G3")
early_features <- setdiff(all_features, c("G1", "G2"))

# ---- 3. Train / test split (75 / 25, matches the Python analysis) -----------

train_idx <- createDataPartition(dataload$G3, p = 0.75, list = FALSE)
train <- dataload[train_idx, ]
test  <- dataload[-train_idx, ]




# =============================================================================
# 4. REGRESSION -- predicting the continuous final grade (G3)
# =============================================================================

r2 <- function(actual, predicted) {
  1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
}

fit_eval_regression <- function(features, label) {
  form <- as.formula(paste("G3 ~", paste(features, collapse = " + ")))
  
  # ---- GLM ----
  glm_fit  <- glm(form, data = train, family = gaussian())
  glm_pred <- predict(glm_fit, newdata = test, type = "response")
  glm_r2   <- r2(test$G3, glm_pred)
  
  # ---- Random Forest ----
  rf_fit  <- randomForest(form, data = train, ntree = 300)
  rf_pred <- predict(rf_fit, newdata = test)
  rf_r2   <- r2(test$G3, rf_pred)
  
  # ---- XGBoost ----
  x_train <- model.matrix(form, data = train)[, -1]  # drop intercept column
  x_test  <- model.matrix(form, data = test)[, -1]
  y_train <- train$G3
  
  xgb_fit <- xgboost(
    data = x_train, label = y_train,
    nrounds = 100, objective = "reg:squarederror",
    verbose = 0
  )
  xgb_pred <- predict(xgb_fit, newdata = x_test)
  xgb_r2   <- r2(test$G3, xgb_pred)
  
  cat(sprintf(
    "%s: GLM R2 = %.3f | Random Forest R2 = %.3f | XGBoost R2 = %.3f\n",
    label, glm_r2, rf_r2, xgb_r2
  ))
  
  list(
    glm_r2 = glm_r2, rf_r2 = rf_r2, xgb_r2 = xgb_r2,
    glm_fit = glm_fit, rf_fit = rf_fit, xgb_fit = xgb_fit
  )
}

cat("\n--- Regression: predicting final grade G3 ---\n")
reg_full  <- fit_eval_regression(all_features,   "With G1/G2 (in-progress prediction)")
reg_early <- fit_eval_regression(early_features, "Without G1/G2 (early prediction)")


# =============================================================================
# 5. Cross Validation -- predicting pass (G3 >= 10) vs. fail
# =============================================================================

r2 <- function(actual, predicted) {
  1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
}

fit_eval_regression <- function(features, label, k = 5) {
  form <- as.formula(paste("G3 ~", paste(features, collapse = " + ")))
  
  ctrl <- trainControl(method = "cv", number = k)
  
  # ---- Cross-validated models (via caret::train, k-fold on train) ----
  glm_cv <- train(form, data = train, method = "glm", family = gaussian(),
                  trControl = ctrl)
  rf_cv  <- train(form, data = train, method = "rf",
                  trControl = ctrl, ntree = 300)
  xgb_cv <- train(form, data = train, method = "xgbTree",
                  trControl = ctrl, verbosity = 0)
  
  get_cv_r2 <- function(model) model$results$Rsquared[which.min(model$results$RMSE)]
  
  glm_cv_r2 <- get_cv_r2(glm_cv)
  rf_cv_r2  <- get_cv_r2(rf_cv)
  xgb_cv_r2 <- get_cv_r2(xgb_cv)
  
  # ---- Fit final models on full train, evaluate on held-out test set ----
  glm_fit  <- glm(form, data = train, family = gaussian())
  glm_pred <- predict(glm_fit, newdata = test, type = "response")
  glm_r2   <- r2(test$G3, glm_pred)
  
  rf_fit  <- randomForest(form, data = train, ntree = 300)
  rf_pred <- predict(rf_fit, newdata = test)
  rf_r2   <- r2(test$G3, rf_pred)
  
  x_train <- model.matrix(form, data = train)[, -1]
  x_test  <- model.matrix(form, data = test)[, -1]
  y_train <- train$G3
  
  xgb_fit <- xgboost(data = x_train, label = y_train,
                     nrounds = 100, objective = "reg:squarederror", verbose = 0)
  xgb_pred <- predict(xgb_fit, newdata = x_test)
  xgb_r2   <- r2(test$G3, xgb_pred)
  
  cat(sprintf("\n%s\n", label))
  cat(sprintf("  [Test set]   GLM=%.3f | RF=%.3f | XGB=%.3f\n",
              glm_r2, rf_r2, xgb_r2))
  cat(sprintf("  [%d-fold CV] GLM=%.3f | RF=%.3f | XGB=%.3f\n",
              k, glm_cv_r2, rf_cv_r2, xgb_cv_r2))
  
  list(
    glm_r2 = glm_r2, rf_r2 = rf_r2, xgb_r2 = xgb_r2,
    glm_cv_r2 = glm_cv_r2, rf_cv_r2 = rf_cv_r2, xgb_cv_r2 = xgb_cv_r2,
    glm_fit = glm_fit, rf_fit = rf_fit, xgb_fit = xgb_fit,
    glm_cv = glm_cv, rf_cv = rf_cv, xgb_cv = xgb_cv
  )
}

cat("\n--- Regression: predicting final grade G3 ---\n")
reg_full  <- fit_eval_regression(all_features,   "With G1/G2 (in-progress prediction)")
reg_early <- fit_eval_regression(early_features, "Without G1/G2 (early prediction)")


# =============================================================================
# 6. CLASSIFICATION -- predicting pass (G3 >= 10) vs. fail
# =============================================================================

dataload$pass    <- factor(ifelse(dataload$G3 >= 10, "pass", "fail"), levels = c("fail", "pass"))
train$pass <- dataload$pass[train_idx]
test$pass  <- dataload$pass[-train_idx]

cat("\nClass balance (full dataset):\n")
print(table(dataload$pass))



fit_eval_classification <- function(features, label) {
  form <- as.formula(paste("pass ~", paste(features, collapse = " + ")))

  rf_fit <- randomForest(form, data = train, ntree = 300)
  pred_class <- predict(rf_fit, newdata = test, type = "class")
  pred_prob  <- predict(rf_fit, newdata = test, type = "prob")[, "pass"]

  acc <- mean(pred_class == test$pass)
  cm  <- confusionMatrix(pred_class, test$pass, positive = "pass")
  f1  <- cm$byClass["F1"]
  auc_val <- as.numeric(auc(roc(test$pass, pred_prob, levels = c("fail", "pass"),
                                 direction = "<", quiet = TRUE)))

  cat(sprintf("%s: Accuracy = %.3f | F1 = %.3f | ROC-AUC = %.3f\n",
              label, acc, f1, auc_val))

  list(rf_fit = rf_fit, acc = acc, f1 = f1, auc = auc_val)
}

cat("\n--- Classification: pass/fail prediction ---\n")
cls_full  <- fit_eval_classification(all_features,   "With G1/G2")
cls_early <- fit_eval_classification(early_features, "Without G1/G2 (early warning)")


# =============================================================================
# 7. FEATURE IMPORTANCE -- early-warning classifier only (no grade leakage)
# =============================================================================

importance_dataload <- importance(cls_early$rf_fit)
importance_dataload <- data.frame(feature = rownames(importance_dataload),
                             MeanDecreaseGini = importance_dataload[, "MeanDecreaseGini"])
importance_dataload <- importance_dataload[order(-importance_dataload$MeanDecreaseGini), ]

cat("\nTop 10 feature importances (early-warning classifier, no G1/G2):\n")
print(head(importance_dataload, 10))

vip(cls_early$rf_fit, num_features = 10)

# =============================================================================
# 8. DESCRIPTIVE / GROUP-LEVEL ANALYSIS
# =============================================================================


# ---- 1. Mean G3 by number of past class failures ----
cat("\nMean G3 by number of past class failures:\n")
by_failures <- aggregate(G3 ~ failures, data = dataload, FUN = function(x) c(mean = mean(x), n = length(x)))
by_failures <- data.frame(failures = by_failures$failures,
                          mean_G3 = by_failures$G3[, "mean"],
                          n = by_failures$G3[, "n"])
print(by_failures)

p1 <- ggplot(by_failures, aes(x = factor(failures), y = mean_G3)) +
  geom_col(fill = "#d95f02") +
  geom_text(aes(label = sprintf("n=%d", n)), vjust = -0.5, size = 3.5) +
  labs(title = "Mean Final Grade (G3) by Number of Past Failures",
       x = "Number of Past Class Failures", y = "Mean G3") +
  ylim(0, max(by_failures$mean_G3) * 1.15) +
  theme_minimal(base_size = 13)

print(p1)

# ---- 2. Mean G3 by weekly study time category ----
cat("\nMean G3 by weekly study time category (1=<2h, 2=2-5h, 3=5-10h, 4=>10h):\n")
by_studytime <- aggregate(G3 ~ studytime, data = dataload, FUN = function(x) c(mean = mean(x), n = length(x)))
by_studytime <- data.frame(studytime = by_studytime$studytime,
                           mean_G3 = by_studytime$G3[, "mean"],
                           n = by_studytime$G3[, "n"])
print(by_studytime)

study_labels <- c("1" = "<2h", "2" = "2-5h", "3" = "5-10h", "4" = ">10h")

p2 <- ggplot(by_studytime, aes(x = factor(studytime, labels = study_labels), y = mean_G3)) +
  geom_col(fill = "#1b9e77") +
  geom_text(aes(label = sprintf("n=%d", n)), vjust = -0.5, size = 3.5) +
  labs(title = "Mean Final Grade (G3) by Weekly Study Time",
       x = "Weekly Study Time", y = "Mean G3") +
  ylim(0, max(by_studytime$mean_G3) * 1.15) +
  theme_minimal(base_size = 13)

print(p2)

# ---- 3. Correlation of numeric predictors with G3 ----
cat("\nCorrelation of numeric predictors with G3:\n")
numeric_vars <- c("studytime", "absences", "failures", "goout", "Dalc", "Walc",
                  "health", "Medu", "Fedu", "G1", "G2", "G3")
corr_matrix <- cor(dataload[, numeric_vars])
corr_with_G3 <- sort(corr_matrix[, "G3"])
print(corr_with_G3)

corr_df <- data.frame(variable = names(corr_with_G3), correlation = corr_with_G3)
corr_df <- corr_df[corr_df$variable != "G3", ]  # drop G3-vs-G3 (always 1.0)

p3 <- ggplot(corr_df, aes(x = reorder(variable, correlation), y = correlation,
                          fill = correlation > 0)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#1b9e77", "FALSE" = "#d95f02"), guide = "none") +
  labs(title = "Correlation of Numeric Predictors with Final Grade (G3)",
       x = "Predictor", y = "Correlation with G3") +
  theme_minimal(base_size = 13)

print(p3)


