# ==============================================================================
# MIS41120: Statistical Learning - Practical Data Analysis Assignment
# Script: stl_Ellora_HirudayaKumar_Le_boston.R
# Purpose: All models applied to the Boston dataset (response: crim).
#          ISLR Chapter 3, Question 15(b) — predict per capita crime rate.
# Depends: stl_Ellora_HirudayaKumar_Le_utils.R (must be in same folder)
# ==============================================================================

source("stl_Ellora_HirudayaKumar_Le_utils.R")
set.seed(123)

# ------------------------------------------------------------------------------
# 1. Load Boston dataset
# ------------------------------------------------------------------------------
# ISLR Q15 (p.128): predict crim (per capita crime rate) using all predictors.
# Dataset: 506 observations, 13 predictors, 1 response (crim).
# ------------------------------------------------------------------------------

data(Boston)
boston <- Boston
cat("Boston dataset dimensions:", dim(boston), "\n")
str(boston)

# ------------------------------------------------------------------------------
# 2. Train/test split (80/20) — done BEFORE any model fitting to prevent leakage
# ------------------------------------------------------------------------------

set.seed(123)
train_idx_b  <- sample(1:nrow(boston), 0.8 * nrow(boston))
boston_train <- boston[train_idx_b, ]
boston_test  <- boston[-train_idx_b, ]

x_boston  <- model.matrix(crim ~ ., boston)[, -1]
y_boston  <- boston$crim
x_train_b <- x_boston[train_idx_b, ]
x_test_b  <- x_boston[-train_idx_b, ]
y_train_b <- y_boston[train_idx_b]
y_test_b  <- y_boston[-train_idx_b]

# ------------------------------------------------------------------------------
# 3. OLS Regression
# Fitted on training set only. Tested on held-out test set.
# Identifies predictors for which H0: beta_j = 0 can be rejected (p < 0.05).
# ------------------------------------------------------------------------------

t0          <- proc.time()
ols_boston  <- lm(crim ~ ., data = boston_train)
time_ols_b  <- (proc.time() - t0)[["elapsed"]]
ols_summary <- summary(ols_boston)
print(ols_summary)

sig_b <- rownames(ols_summary$coefficients)[
  ols_summary$coefficients[, "Pr(>|t|)"] < 0.05
]
cat("\nBoston OLS - Significant predictors (p < 0.05):\n"); print(sig_b)

ols_pred_b <- predict(ols_boston, newdata = boston_test)
ols_mse_b  <- mean((ols_pred_b - y_test_b)^2)
cat("Boston OLS - Test MSE:", ols_mse_b, "\n")

# ------------------------------------------------------------------------------
# 4. Ridge, Lasso, Elastic Net via run_glmnet() helper
# 10-fold CV selects optimal lambda for each. alpha: 0=Ridge, 1=Lasso, 0.5=EN.
# ------------------------------------------------------------------------------

ridge_b <- run_glmnet(x_train_b, y_train_b, x_test_b, y_test_b,
                       alpha = 0, label = "Boston Ridge")
lasso_b <- run_glmnet(x_train_b, y_train_b, x_test_b, y_test_b,
                       alpha = 1, label = "Boston Lasso")
enet_b  <- run_glmnet(x_train_b, y_train_b, x_test_b, y_test_b,
                       alpha = 0.5, label = "Boston Elastic Net")

# Store results for convenience
ridge_mse_b <- ridge_b$mse; time_ridge_b <- ridge_b$time_sec
lasso_mse_b <- lasso_b$mse; time_lasso_b <- lasso_b$time_sec
enet_mse_b  <- enet_b$mse;  time_enet_b  <- enet_b$time_sec

# Store fitted objects for plots
cv_ridge_b        <- ridge_b$fit; best_lambda_ridge_b <- ridge_b$lambda
cv_lasso_b        <- lasso_b$fit; best_lambda_lasso_b <- lasso_b$lambda
cv_enet_b         <- enet_b$fit;  best_lambda_enet_b  <- enet_b$lambda

# ------------------------------------------------------------------------------
# 5. SVM Regression
# ------------------------------------------------------------------------------
# Per course notes: SVM objective = hinge loss + (1/(2C)) * ||w||^2.
# Regularisation strength = 1/(2C):
#   large C -> 1/(2C) -> 0 -> effectively no regularisation
#   small C -> strong L2 regularisation
# C=1000 approximates the unregularised baseline (1/2000 ~ 0).
# CV over C_grid selects the best-regularised model.
# ------------------------------------------------------------------------------

train_df_b <- data.frame(x_train_b, crim = y_train_b)
test_df_b  <- data.frame(x_test_b,  crim = y_test_b)

set.seed(123)
svm_tune_b <- tune(svm, crim ~ ., data = train_df_b, kernel = "radial",
                   ranges = list(cost = C_grid),
                   tunecontrol = tune.control(cross = 10))
cat("\nBoston SVM - 10-fold CV across C grid:\n")
print(svm_tune_b$performances)

svm_unreg_b_res <- run_svm_reg(train_df_b, test_df_b, "crim",
                                cost = 1000, label = "Boston SVM (C=1000, ~unregularised)")
svm_l2_b_res    <- run_svm_reg(train_df_b, test_df_b, "crim",
                                cost = svm_tune_b$best.parameters$cost,
                                label = paste("Boston SVM (CV-best C =",
                                              svm_tune_b$best.parameters$cost, ", L2)"))

svm_mse_unreg_b <- svm_unreg_b_res$mse; time_svm_unreg_b <- svm_unreg_b_res$time_sec
svm_mse_l2_b    <- svm_l2_b_res$mse;    time_svm_l2_b    <- svm_l2_b_res$time_sec
best_C_b        <- svm_tune_b$best.parameters$cost

# ------------------------------------------------------------------------------
# 5b. SVM Classification (for L1 / Elastic Net via sparseSVM)
# ------------------------------------------------------------------------------
# e1071 supports only L2 regularisation via C. For L1 and Elastic Net SVMs,
# sparseSVM requires binary classification. Threshold from training median only
# to avoid test-set leakage.
# ------------------------------------------------------------------------------

median_crim_train <- median(y_train_b)
y_train_b_class   <- as.factor(ifelse(y_train_b > median_crim_train, 1, 0))
y_test_b_class    <- as.factor(ifelse(y_test_b  > median_crim_train, 1, 0))
train_df_b_class  <- data.frame(x_train_b, class = y_train_b_class)
test_df_b_class   <- data.frame(x_test_b,  class = y_test_b_class)

run_svm_cls(train_df_b_class, test_df_b_class, cost = 1000,
            label = "Boston SVM Classification (C=1000, ~unregularised)")

set.seed(123)
svm_tune_b_cl    <- tune(svm, class ~ ., data = train_df_b_class, kernel = "radial",
                          ranges = list(cost = C_grid),
                          tunecontrol = tune.control(cross = 10))
best_C_b_cl      <- svm_tune_b_cl$best.parameters$cost
run_svm_cls(train_df_b_class, test_df_b_class, cost = best_C_b_cl,
            label = paste("Boston SVM Classification (CV-best C =", best_C_b_cl, ", L2)"))

y_train_b_num <- ifelse(y_train_b_class == "1", 1, -1)
y_test_b_num  <- ifelse(y_test_b_class  == "1", 1, -1)

sparse_lasso_b <- cv.sparseSVM(x_train_b, y_train_b_num, alpha = 1)
pred_lasso_b   <- predict(sparse_lasso_b, x_test_b)
acc_lasso_b    <- mean(pred_lasso_b == y_test_b_num)
cat("Boston Sparse SVM (L1) - Accuracy:", acc_lasso_b, "\n")
print(table(Predicted = pred_lasso_b, Actual = y_test_b_num))

sparse_enet_b <- cv.sparseSVM(x_train_b, y_train_b_num, alpha = 0.5)
pred_enet_b   <- predict(sparse_enet_b, x_test_b)
acc_enet_b    <- mean(pred_enet_b == y_test_b_num)
cat("Boston Sparse SVM (Elastic Net) - Accuracy:", acc_enet_b, "\n")
print(table(Predicted = pred_enet_b, Actual = y_test_b_num))

# ------------------------------------------------------------------------------
# 6. MLP (Multilayer Perceptron)
# ------------------------------------------------------------------------------
# Normalisation computed from TRAINING data only (col_min_b, col_max_b)
# then applied to test set — prevents data leakage.
# neuralnet: unregularised baseline (manageable at p=13)
# nnet:      L2 via weight decay parameter
# ANN2:      L1 and Elastic Net (cited in assignment brief Sec 2.1)
# tanh activation used for ANN2: crim is right-skewed; tanh is more
# numerically stable than relu on skewed normalised targets.
# ------------------------------------------------------------------------------

col_min_b    <- apply(boston_train, 2, min)
col_max_b    <- apply(boston_train, 2, max)
train_norm_b <- normalize_df(boston_train, col_min_b, col_max_b)
test_norm_b  <- normalize_df(boston_test,  col_min_b, col_max_b)

crim_min_b <- col_min_b["crim"]; crim_max_b <- col_max_b["crim"]
rescale_b  <- function(x) x * (crim_max_b - crim_min_b) + crim_min_b

predictors_b  <- names(boston)[names(boston) != "crim"]
mlp_formula_b <- as.formula(paste("crim ~", paste(predictors_b, collapse = " + ")))

# Unregularised MLP (neuralnet, hidden layers 5-3, no penalty)
set.seed(123)
t0               <- proc.time()
mlp_unreg_b      <- neuralnet(mlp_formula_b, data = train_norm_b,
                               hidden = c(5, 3), linear.output = TRUE)
time_mlp_unreg_b <- (proc.time() - t0)[["elapsed"]]
mlp_pred_unreg_b <- predict(mlp_unreg_b, test_norm_b)
mlp_mse_unreg_b  <- mean((rescale_b(mlp_pred_unreg_b) - boston_test$crim)^2)
cat("\nBoston MLP (unregularised, neuralnet) - Test MSE:", mlp_mse_unreg_b, "\n")

# L2-regularised MLP (nnet, weight decay = L2 penalty on weights)
set.seed(123)
t0            <- proc.time()
mlp_l2_b      <- nnet(mlp_formula_b, data = train_norm_b, size = 5, decay = 0.01,
                       linout = TRUE, trace = FALSE, maxit = 500)
time_mlp_l2_b <- (proc.time() - t0)[["elapsed"]]
mlp_pred_l2_b <- predict(mlp_l2_b, test_norm_b)
mlp_mse_l2_b  <- mean((rescale_b(mlp_pred_l2_b) - boston_test$crim)^2)
cat("Boston MLP (L2 weight decay=0.01, nnet) - Test MSE:", mlp_mse_l2_b, "\n")

# L1-regularised MLP (ANN2, L1 penalty)
x_train_b_mat <- as.matrix(train_norm_b[, predictors_b])
y_train_b_mat <- as.matrix(train_norm_b[, "crim"])
x_test_b_mat  <- as.matrix(test_norm_b[, predictors_b])

set.seed(123)
t0            <- proc.time()
mlp_l1_b      <- neuralnetwork(X = x_train_b_mat, y = y_train_b_mat,
                                hidden.layers = c(5, 3), regression = TRUE,
                                loss.type = "squared", activ.functions = "tanh",
                                L1 = 0.01, L2 = 0,
                                n.epochs = 500, learn.rates = 0.001,
                                val.prop = 0, verbose = FALSE)
time_mlp_l1_b <- (proc.time() - t0)[["elapsed"]]
mlp_pred_l1_b <- predict(mlp_l1_b, newdata = x_test_b_mat)$predictions
mlp_mse_l1_b  <- mean((rescale_b(mlp_pred_l1_b) - boston_test$crim)^2)
cat("Boston MLP (L1 / Lasso, ANN2) - Test MSE:", mlp_mse_l1_b, "\n")

# Elastic Net MLP (ANN2, combined L1 + L2)
set.seed(123)
t0              <- proc.time()
mlp_enet_b      <- neuralnetwork(X = x_train_b_mat, y = y_train_b_mat,
                                  hidden.layers = c(5, 3), regression = TRUE,
                                  loss.type = "squared", activ.functions = "tanh",
                                  L1 = 0.005, L2 = 0.005,
                                  n.epochs = 500, learn.rates = 0.001,
                                  val.prop = 0, verbose = FALSE)
time_mlp_enet_b  <- (proc.time() - t0)[["elapsed"]]
mlp_pred_enet_b  <- predict(mlp_enet_b, newdata = x_test_b_mat)$predictions
mlp_mse_enet_b   <- mean((rescale_b(mlp_pred_enet_b) - boston_test$crim)^2)
cat("Boston MLP (Elastic Net, ANN2) - Test MSE:", mlp_mse_enet_b, "\n")

# ------------------------------------------------------------------------------
# 7. Residual diagnostic plots for best regression models
# ------------------------------------------------------------------------------
# Best-performing model per family saved as PNG for inclusion in the report.
# Predicted vs Actual: good model -> points on the diagonal.
# Residuals vs Fitted: good model -> random scatter around zero.
# ------------------------------------------------------------------------------

residual_plots(boston_test$crim, ols_pred_b,
               "Boston OLS", "stl_Ellora_HirudayaKumar_Le_residuals_boston_ols.png")

residual_plots(boston_test$crim, as.numeric(lasso_b$predictions),
               "Boston Lasso (L1)", "stl_Ellora_HirudayaKumar_Le_residuals_boston_lasso.png")

residual_plots(boston_test$crim, svm_l2_b_res$predictions,
               "Boston SVM (CV-best C)", "stl_Ellora_HirudayaKumar_Le_residuals_boston_svm.png")

residual_plots(boston_test$crim, rescale_b(mlp_pred_unreg_b),
               "Boston MLP (unregularised)", "stl_Ellora_HirudayaKumar_Le_residuals_boston_mlp.png")

cat("\nBoston script completed successfully.\n")

# ==============================================================================
# END OF BOSTON SCRIPT
# ==============================================================================
