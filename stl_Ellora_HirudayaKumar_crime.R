# ==============================================================================
# MIS41120: Statistical Learning - Practical Data Analysis Assignment
# Script: stl_Ellora_HirudayaKumar_Le_crime.R
# Purpose: All models applied to the Communities and Crime dataset.
#          Response: ViolentCrimesPerPop (continuous, range [0,1])
#          Source: https://archive.ics.uci.edu/dataset/183/communities+and+crime
#          Place communities.data in the same folder as this script.
# Depends: stl_Ellora_HirudayaKumar_Le_utils.R (must be in same folder)
# ==============================================================================

source("stl_Ellora_HirudayaKumar_Le_utils.R")
set.seed(123)

# ------------------------------------------------------------------------------
# 1. Load and clean Communities and Crime dataset
# ------------------------------------------------------------------------------
# Raw file has no header. First 5 columns are non-predictive identifiers
# (state, county, community, communityname, fold) dropped per UCI documentation.
# Columns with >= 50% missing values removed (mainly policing-related fields).
# Remaining rows with any NA dropped via na.omit().
# After cleaning: 1993 observations, 100 predictors, 1 response.
# ------------------------------------------------------------------------------

crime_raw <- read.csv("communities.data", header = FALSE, na.strings = "?")
crime     <- crime_raw[, -c(1:5)]

na_proportion <- colSums(is.na(crime)) / nrow(crime)
crime <- crime[, na_proportion < 0.5]
crime <- na.omit(crime)
colnames(crime)[ncol(crime)] <- "ViolentCrimesPerPop"

cat("Communities and Crime - dimensions after cleaning:", dim(crime), "\n")
cat("Predictors retained:", ncol(crime) - 1, "\n")
cat("Observations retained:", nrow(crime), "\n")

# ------------------------------------------------------------------------------
# 2. Train/test split (80/20) — done BEFORE any model fitting
# ------------------------------------------------------------------------------

set.seed(123)
train_idx_c <- sample(1:nrow(crime), 0.8 * nrow(crime))
crime_train <- crime[train_idx_c, ]
crime_test  <- crime[-train_idx_c, ]

x_crime   <- model.matrix(ViolentCrimesPerPop ~ ., crime)[, -1]
y_crime   <- crime$ViolentCrimesPerPop
x_train_c <- x_crime[train_idx_c, ]
x_test_c  <- x_crime[-train_idx_c, ]
y_train_c <- y_crime[train_idx_c]
y_test_c  <- y_crime[-train_idx_c]

# ------------------------------------------------------------------------------
# 3. OLS Regression — fitted on training data only
# ------------------------------------------------------------------------------

t0                <- proc.time()
ols_crime         <- lm(ViolentCrimesPerPop ~ ., data = crime_train)
time_ols_c        <- (proc.time() - t0)[["elapsed"]]
ols_crime_summary <- summary(ols_crime)

sig_c <- rownames(ols_crime_summary$coefficients)[
  ols_crime_summary$coefficients[, "Pr(>|t|)"] < 0.05
]
cat("\nCommunities and Crime OLS - Significant predictors (p < 0.05):\n")
print(sig_c)
cat("Communities and Crime OLS - R-squared:", ols_crime_summary$r.squared, "\n")

ols_pred_c <- predict(ols_crime, newdata = crime_test)
ols_mse_c  <- mean((ols_pred_c - y_test_c)^2)
cat("Communities and Crime OLS - Test MSE:", ols_mse_c, "\n")

# ------------------------------------------------------------------------------
# 4. Ridge, Lasso, Elastic Net via run_glmnet() helper
# With 100 predictors, Lasso is expected to eliminate many via variable selection.
# ------------------------------------------------------------------------------

ridge_c <- run_glmnet(x_train_c, y_train_c, x_test_c, y_test_c,
                       alpha = 0, label = "Communities and Crime Ridge")
lasso_c <- run_glmnet(x_train_c, y_train_c, x_test_c, y_test_c,
                       alpha = 1, label = "Communities and Crime Lasso")
enet_c  <- run_glmnet(x_train_c, y_train_c, x_test_c, y_test_c,
                       alpha = 0.5, label = "Communities and Crime Elastic Net")

ridge_mse_c <- ridge_c$mse; time_ridge_c <- ridge_c$time_sec
lasso_mse_c <- lasso_c$mse; time_lasso_c <- lasso_c$time_sec
enet_mse_c  <- enet_c$mse;  time_enet_c  <- enet_c$time_sec

cv_ridge_c        <- ridge_c$fit; best_lambda_ridge_c <- ridge_c$lambda
cv_lasso_c        <- lasso_c$fit; best_lambda_lasso_c <- lasso_c$lambda
cv_enet_c         <- enet_c$fit;  best_lambda_enet_c  <- enet_c$lambda

# ------------------------------------------------------------------------------
# 5. SVM Regression
# ------------------------------------------------------------------------------

train_df_c <- data.frame(x_train_c, ViolentCrimesPerPop = y_train_c)
test_df_c  <- data.frame(x_test_c,  ViolentCrimesPerPop = y_test_c)

set.seed(123)
svm_tune_c <- tune(svm, ViolentCrimesPerPop ~ ., data = train_df_c,
                   kernel = "radial", ranges = list(cost = C_grid),
                   tunecontrol = tune.control(cross = 10))
cat("\nCommunities and Crime SVM - 10-fold CV across C grid:\n")
print(svm_tune_c$performances)

svm_unreg_c_res <- run_svm_reg(train_df_c, test_df_c, "ViolentCrimesPerPop",
                                cost = 1000,
                                label = "Communities and Crime SVM (C=1000, ~unregularised)")
svm_l2_c_res    <- run_svm_reg(train_df_c, test_df_c, "ViolentCrimesPerPop",
                                cost = svm_tune_c$best.parameters$cost,
                                label = paste("Communities and Crime SVM (CV-best C =",
                                              svm_tune_c$best.parameters$cost, ", L2)"))

svm_mse_unreg_c <- svm_unreg_c_res$mse; time_svm_unreg_c <- svm_unreg_c_res$time_sec
svm_mse_l2_c    <- svm_l2_c_res$mse;    time_svm_l2_c    <- svm_l2_c_res$time_sec
best_C_c        <- svm_tune_c$best.parameters$cost

# ------------------------------------------------------------------------------
# 5b. SVM Classification (for L1 / Elastic Net via sparseSVM)
# Threshold computed from training median only to avoid leakage.
# ------------------------------------------------------------------------------

median_crime_train <- median(y_train_c)
y_train_c_class    <- as.factor(ifelse(y_train_c > median_crime_train, 1, 0))
y_test_c_class     <- as.factor(ifelse(y_test_c  > median_crime_train, 1, 0))
train_df_c_class   <- data.frame(x_train_c, class = y_train_c_class)
test_df_c_class    <- data.frame(x_test_c,  class = y_test_c_class)

run_svm_cls(train_df_c_class, test_df_c_class, cost = 1000,
            label = "Communities and Crime SVM Classification (C=1000, ~unregularised)")

set.seed(123)
svm_tune_c_cl <- tune(svm, class ~ ., data = train_df_c_class, kernel = "radial",
                       ranges = list(cost = C_grid),
                       tunecontrol = tune.control(cross = 10))
best_C_c_cl   <- svm_tune_c_cl$best.parameters$cost
run_svm_cls(train_df_c_class, test_df_c_class, cost = best_C_c_cl,
            label = paste("Communities and Crime SVM Classification (CV-best C =",
                          best_C_c_cl, ", L2)"))

y_train_c_num <- ifelse(y_train_c_class == "1", 1, -1)
y_test_c_num  <- ifelse(y_test_c_class  == "1", 1, -1)

sparse_lasso_c <- cv.sparseSVM(x_train_c, y_train_c_num, alpha = 1)
pred_lasso_c   <- predict(sparse_lasso_c, x_test_c)
acc_lasso_c    <- mean(pred_lasso_c == y_test_c_num)
cat("Communities and Crime Sparse SVM (L1) - Accuracy:", acc_lasso_c, "\n")
print(table(Predicted = pred_lasso_c, Actual = y_test_c_num))

sparse_enet_c <- cv.sparseSVM(x_train_c, y_train_c_num, alpha = 0.5)
pred_enet_c   <- predict(sparse_enet_c, x_test_c)
acc_enet_c    <- mean(pred_enet_c == y_test_c_num)
cat("Communities and Crime Sparse SVM (Elastic Net) - Accuracy:", acc_enet_c, "\n")
print(table(Predicted = pred_enet_c, Actual = y_test_c_num))

# ------------------------------------------------------------------------------
# 6. MLP (Multilayer Perceptron)
# ------------------------------------------------------------------------------
# Normalisation from training data only to prevent leakage.
# nnet used for unregularised and L2 (neuralnet fails on p=100).
# 5 random restarts for nnet: high-dimensional data with no regularisation
# is prone to local minima; best of 5 runs is retained.
# The unregularised MLP is expected to overfit severely at p=100 — this
# is a genuine finding demonstrating the necessity of regularisation in
# high-dimensional settings (discussed in the report).
# ANN2 used for L1 and Elastic Net as cited in assignment brief (Sec 2.1).
# MSE computed on normalised [0,1] scale since response is already normalised
# after min-max scaling and all comparison values are on this same scale.
# ------------------------------------------------------------------------------

col_min_c    <- apply(crime_train, 2, min)
col_max_c    <- apply(crime_train, 2, max)
train_norm_c <- normalize_df(crime_train, col_min_c, col_max_c)
test_norm_c  <- normalize_df(crime_test,  col_min_c, col_max_c)

y_test_c_norm <- test_norm_c$ViolentCrimesPerPop
predictors_c  <- names(crime)[names(crime) != "ViolentCrimesPerPop"]
mlp_formula_c <- as.formula(paste("ViolentCrimesPerPop ~",
                                   paste(predictors_c, collapse = " + ")))

# Unregularised MLP (nnet, decay=0, 5 restarts)
t0 <- proc.time()
best_mlp_unreg_c <- NULL; best_val_unreg_c <- Inf
for (i in 1:5) {
  set.seed(i * 10)
  tmp <- nnet(mlp_formula_c, data = train_norm_c, size = 10, decay = 0,
              linout = TRUE, trace = FALSE, maxit = 2000, MaxNWts = 2000)
  if (tmp$value < best_val_unreg_c) {
    best_val_unreg_c <- tmp$value; best_mlp_unreg_c <- tmp
  }
}
time_mlp_unreg_c <- (proc.time() - t0)[["elapsed"]]
mlp_pred_unreg_c <- predict(best_mlp_unreg_c, test_norm_c)
mlp_mse_unreg_c  <- mean((mlp_pred_unreg_c - y_test_c_norm)^2)
cat("\nCommunities and Crime MLP (unregularised, decay=0) - Test MSE:",
    mlp_mse_unreg_c, "\n")
cat("NOTE: High MSE expected here — severe overfitting at p=100 without",
    "regularisation.\n")

# L2-regularised MLP (nnet, weight decay=0.01, 5 restarts)
t0 <- proc.time()
best_mlp_l2_c <- NULL; best_val_l2_c <- Inf
for (i in 1:5) {
  set.seed(i * 10)
  tmp <- nnet(mlp_formula_c, data = train_norm_c, size = 10, decay = 0.01,
              linout = TRUE, trace = FALSE, maxit = 2000, MaxNWts = 2000)
  if (tmp$value < best_val_l2_c) {
    best_val_l2_c <- tmp$value; best_mlp_l2_c <- tmp
  }
}
time_mlp_l2_c <- (proc.time() - t0)[["elapsed"]]
mlp_pred_l2_c <- predict(best_mlp_l2_c, test_norm_c)
mlp_mse_l2_c  <- mean((mlp_pred_l2_c - y_test_c_norm)^2)
cat("Communities and Crime MLP (L2 decay=0.01) - Test MSE:", mlp_mse_l2_c, "\n")

# L1-regularised MLP (ANN2, lambda1=L1 penalty)
x_train_c_mat <- as.matrix(train_norm_c[, predictors_c])
y_train_c_mat <- as.matrix(train_norm_c[, "ViolentCrimesPerPop"])
x_test_c_mat  <- as.matrix(test_norm_c[, predictors_c])

set.seed(123)
t0            <- proc.time()
mlp_l1_c      <- neuralnetwork(X = x_train_c_mat, y = y_train_c_mat,
                                hidden.layers = c(10, 5), regression = TRUE,
                                loss.type = "squared", activ.functions = "tanh",
                                L1 = 0.01, L2 = 0,
                                n.epochs = 500, learn.rates = 0.001,
                                val.prop = 0, verbose = FALSE)
time_mlp_l1_c <- (proc.time() - t0)[["elapsed"]]
mlp_pred_l1_c <- predict(mlp_l1_c, newdata = x_test_c_mat)$predictions
mlp_mse_l1_c  <- mean((mlp_pred_l1_c - y_test_c_norm)^2)
cat("Communities and Crime MLP (L1 / Lasso, ANN2) - Test MSE:", mlp_mse_l1_c, "\n")

# Elastic Net MLP (ANN2, combined L1 + L2)
set.seed(123)
t0              <- proc.time()
mlp_enet_c      <- neuralnetwork(X = x_train_c_mat, y = y_train_c_mat,
                                  hidden.layers = c(10, 5), regression = TRUE,
                                  loss.type = "squared", activ.functions = "tanh",
                                  L1 = 0.005, L2 = 0.005,
                                  n.epochs = 500, learn.rates = 0.001,
                                  val.prop = 0, verbose = FALSE)
time_mlp_enet_c  <- (proc.time() - t0)[["elapsed"]]
mlp_pred_enet_c  <- predict(mlp_enet_c, newdata = x_test_c_mat)$predictions
mlp_mse_enet_c   <- mean((mlp_pred_enet_c - y_test_c_norm)^2)
cat("Communities and Crime MLP (Elastic Net, ANN2) - Test MSE:", mlp_mse_enet_c, "\n")

# ------------------------------------------------------------------------------
# 7. Residual diagnostic plots
# ------------------------------------------------------------------------------

residual_plots(y_test_c, ols_pred_c,
               "Communities and Crime OLS", "stl_Ellora_HirudayaKumar_Le_residuals_crime_ols.png")

residual_plots(y_test_c, as.numeric(lasso_c$predictions),
               "Communities and Crime Lasso (L1)", "stl_Ellora_HirudayaKumar_Le_residuals_crime_lasso.png")

residual_plots(y_test_c, svm_l2_c_res$predictions,
               "Communities and Crime SVM (CV-best C)", "stl_Ellora_HirudayaKumar_Le_residuals_crime_svm.png")

residual_plots(y_test_c_norm, mlp_pred_l1_c,
               "Communities and Crime MLP L1 (ANN2)", "stl_Ellora_HirudayaKumar_Le_residuals_crime_mlp.png")

cat("\nCommunities and Crime script completed successfully.\n")

# ==============================================================================
# END OF CRIME SCRIPT
# ==============================================================================
