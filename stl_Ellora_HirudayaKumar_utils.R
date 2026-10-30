# ==============================================================================
# MIS41120: Statistical Learning - Practical Data Analysis Assignment
# Script: stl_Ellora_HirudayaKumar_Le_utils.R
# Purpose: Shared utility functions and package setup used by all other scripts.
#          Source this file at the top of each analysis script.
# ==============================================================================

# ------------------------------------------------------------------------------
# Package installation and loading
# ------------------------------------------------------------------------------

required_packages <- c("MASS", "glmnet", "e1071", "neuralnet", "nnet",
                       "caret", "dplyr", "ANN2")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

if (!requireNamespace("sparseSVM", quietly = TRUE)) {
  install.packages("sparseSVM", repos = "https://cran.r-project.org")
}
library(sparseSVM)

# ------------------------------------------------------------------------------
# normalize_df()
# Min-max normalise a data frame using pre-computed training statistics.
# Parameters computed from training data only to prevent data leakage.
# Args:
#   df  : data frame to normalise
#   mn  : named vector of column minimums (from training set)
#   mx  : named vector of column maximums (from training set)
# Returns: normalised data frame
# ------------------------------------------------------------------------------

normalize_df <- function(df, mn, mx) {
  as.data.frame(mapply(function(col, m, x) (col - m) / (x - m),
                       df, mn, mx, SIMPLIFY = FALSE))
}

# ------------------------------------------------------------------------------
# run_glmnet()
# Fits Ridge (alpha=0), Lasso (alpha=1), or Elastic Net (0<alpha<1) via
# 10-fold cross-validation using cv.glmnet. Returns MSE, best lambda,
# predictions, coefficients, and training time.
# Args:
#   x_train, y_train : training matrix and response
#   x_test,  y_test  : test matrix and response
#   alpha            : penalty mixing parameter (0=Ridge, 1=Lasso, 0.5=EN)
#   label            : string label for console output
# Returns: named list with mse, lambda, predictions, coefs, time_sec
# ------------------------------------------------------------------------------

run_glmnet <- function(x_train, y_train, x_test, y_test, alpha, label) {
  t0  <- proc.time()
  fit <- cv.glmnet(x_train, y_train, alpha = alpha, nfolds = 10)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  best_lambda <- fit$lambda.min
  preds       <- as.numeric(predict(fit, s = best_lambda, newx = x_test))
  mse         <- mean((preds - y_test)^2)
  coefs       <- coef(fit, s = best_lambda)
  n_zero      <- sum(coefs == 0) - 1  # exclude intercept

  cat(label, "- Best lambda:", best_lambda, "| Test MSE:", mse,
      "| Predictors eliminated:", n_zero, "\n")

  list(mse = mse, lambda = best_lambda, predictions = preds,
       coefs = coefs, n_eliminated = n_zero, time_sec = elapsed, fit = fit)
}

# ------------------------------------------------------------------------------
# run_svm_reg()
# Fits SVM regression (radial kernel) at a given cost C.
# Args:
#   train_df, test_df : data frames with all predictors and response
#   response          : character name of the response column
#   cost              : C parameter value
#   label             : string label for console output
# Returns: named list with mse, predictions, time_sec
# ------------------------------------------------------------------------------

run_svm_reg <- function(train_df, test_df, response, cost, label) {
  formula <- as.formula(paste(response, "~ ."))
  t0  <- proc.time()
  fit <- svm(formula, data = train_df, kernel = "radial", cost = cost)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  preds <- predict(fit, newdata = test_df)
  y_test <- test_df[[response]]
  mse   <- mean((preds - y_test)^2)
  cat(label, "- Test MSE:", mse, "\n")

  list(mse = mse, predictions = preds, model = fit, time_sec = elapsed)
}

# ------------------------------------------------------------------------------
# run_svm_cls()
# Fits SVM classification (radial kernel) and prints confusion matrix.
# Args:
#   train_df, test_df : data frames with all predictors and class column
#   cost              : C parameter value
#   label             : string label for console output
# Returns: named list with accuracy, confusion matrix
# ------------------------------------------------------------------------------

run_svm_cls <- function(train_df, test_df, cost, label) {
  t0  <- proc.time()
  fit <- svm(class ~ ., data = train_df, kernel = "radial", cost = cost)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  preds <- predict(fit, newdata = test_df)
  acc   <- mean(preds == test_df$class)
  cat(label, "- Accuracy:", acc, "\n")
  cat("Confusion matrix:\n")
  print(table(Predicted = preds, Actual = test_df$class))

  list(accuracy = acc, model = fit, time_sec = elapsed)
}

# ------------------------------------------------------------------------------
# residual_plots()
# Saves predicted vs actual and residual vs fitted plots to PNG.
# Args:
#   actual    : vector of actual response values
#   predicted : vector of predicted values
#   label     : string used in plot titles and filename
#   filename  : output PNG filename (relative path)
# ------------------------------------------------------------------------------

residual_plots <- function(actual, predicted, label, filename) {
  residuals <- actual - predicted

  png(filename, width = 1000, height = 500)
  par(mfrow = c(1, 2), mar = c(5, 4, 4, 2))

  # Predicted vs Actual
  plot(actual, predicted,
       xlab = "Actual", ylab = "Predicted",
       main = paste(label, "\nPredicted vs Actual"),
       pch = 19, col = rgb(0.2, 0.4, 0.8, 0.5), cex = 0.7)
  abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
  legend("topleft", legend = "Perfect fit", col = "red", lty = 2, lwd = 2)

  # Residuals vs Fitted
  plot(predicted, residuals,
       xlab = "Fitted values", ylab = "Residuals",
       main = paste(label, "\nResiduals vs Fitted"),
       pch = 19, col = rgb(0.8, 0.3, 0.2, 0.5), cex = 0.7)
  abline(h = 0, col = "red", lwd = 2, lty = 2)

  dev.off()
  cat("Saved:", filename, "\n")
}

# Shared C grid for SVM cross-validation across all models
C_grid <- c(0.01, 0.1, 1, 10, 100, 1000)

cat("Utils loaded successfully.\n")
