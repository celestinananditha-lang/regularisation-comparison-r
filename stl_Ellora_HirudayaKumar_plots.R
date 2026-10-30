# ==============================================================================
# MIS41120: Statistical Learning - Practical Data Analysis Assignment
# Script: stl_Ellora_HirudayaKumar_Le_plots.R
# Purpose: Complete summary table and all diagnostic plots.
#          Must be run AFTER both boston and crime scripts have completed,
#          as it depends on results objects in the global environment.
# Depends: stl_Ellora_HirudayaKumar_Le_utils.R
#          stl_Ellora_HirudayaKumar_Le_boston.R
#          stl_Ellora_HirudayaKumar_Le_crime.R
# ==============================================================================

source("stl_Ellora_HirudayaKumar_Le_utils.R")
source("stl_Ellora_HirudayaKumar_Le_boston.R")
source("stl_Ellora_HirudayaKumar_Le_crime.R")

# ==============================================================================
# PART 1: COMPLETE SUMMARY TABLE
# ==============================================================================

results <- data.frame(
  Dataset = c(rep("Boston", 10), rep("Communities and Crime", 10)),
  Method  = rep(c("OLS",
                  "Ridge (L2)",
                  "Lasso (L1)",
                  "Elastic Net",
                  "SVM (~unregularised, C=1000)",
                  "SVM (L2-regularised, CV-best C)",
                  "MLP (unregularised)",
                  "MLP (L2 weight decay)",
                  "MLP (L1 / Lasso, ANN2)",
                  "MLP (Elastic Net, ANN2)"), 2),
  Test_MSE = c(
    ols_mse_b, ridge_mse_b, lasso_mse_b, enet_mse_b,
    svm_mse_unreg_b, svm_mse_l2_b,
    mlp_mse_unreg_b, mlp_mse_l2_b, mlp_mse_l1_b, mlp_mse_enet_b,
    ols_mse_c, ridge_mse_c, lasso_mse_c, enet_mse_c,
    svm_mse_unreg_c, svm_mse_l2_c,
    mlp_mse_unreg_c, mlp_mse_l2_c, mlp_mse_l1_c, mlp_mse_enet_c
  ),
  Execution_Time_Sec = c(
    time_ols_b, time_ridge_b, time_lasso_b, time_enet_b,
    time_svm_unreg_b, time_svm_l2_b,
    time_mlp_unreg_b, time_mlp_l2_b, time_mlp_l1_b, time_mlp_enet_b,
    time_ols_c, time_ridge_c, time_lasso_c, time_enet_c,
    time_svm_unreg_c, time_svm_l2_c,
    time_mlp_unreg_c, time_mlp_l2_c, time_mlp_l1_c, time_mlp_enet_c
  )
)

cat("\n\n=== COMPLETE SUMMARY TABLE: Test MSE + Execution Time ===\n")
print(results)
write.csv(results, "stl_Ellora_HirudayaKumar_Le_results_summary.csv", row.names = FALSE)
cat("Results saved to stl_Ellora_HirudayaKumar_Le_results_summary.csv\n")

# ==============================================================================
# PART 2: DIAGNOSTIC PLOTS
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 Lasso Coefficient Shrinkage Path — Boston
# ------------------------------------------------------------------------------

png("stl_Ellora_HirudayaKumar_Le_lasso_path_boston.png", width = 800, height = 600)
plot(cv_lasso_b$glmnet.fit, xvar = "lambda", label = TRUE,
     main = "Boston: Lasso Coefficient Shrinkage Path (response: crim)")
abline(v = log(best_lambda_lasso_b), col = "red", lty = 2)
legend("topright", legend = "CV-optimal lambda", col = "red", lty = 2)
dev.off()
cat("Saved: stl_Ellora_HirudayaKumar_Le_lasso_path_boston.png\n")

# ------------------------------------------------------------------------------
# 2.2 Lasso Coefficient Shrinkage Path — Communities and Crime
# ------------------------------------------------------------------------------

png("stl_Ellora_HirudayaKumar_Le_lasso_path_crime.png", width = 800, height = 600)
plot(cv_lasso_c$glmnet.fit, xvar = "lambda", label = FALSE,
     main = "Communities and Crime: Lasso Shrinkage Path (56/100 predictors eliminated)")
abline(v = log(best_lambda_lasso_c), col = "red", lty = 2)
legend("topright", legend = "CV-optimal lambda", col = "red", lty = 2)
dev.off()
cat("Saved: stl_Ellora_HirudayaKumar_Le_lasso_path_crime.png\n")

# ------------------------------------------------------------------------------
# 2.3 SVM CV Error vs C — Boston
# ------------------------------------------------------------------------------

png("stl_Ellora_HirudayaKumar_Le_svm_cv_boston.png", width = 800, height = 600)
plot(log10(svm_tune_b$performances$cost),
     svm_tune_b$performances$error,
     type = "b", pch = 19, col = "steelblue",
     xlab = "log10(C)", ylab = "10-fold CV MSE",
     main = "Boston SVM: CV Error vs Regularisation Parameter C")
abline(v = log10(best_C_b), col = "red", lty = 2)
legend("topright", legend = paste("Best C =", best_C_b), col = "red", lty = 2)
dev.off()
cat("Saved: stl_Ellora_HirudayaKumar_Le_svm_cv_boston.png\n")

# ------------------------------------------------------------------------------
# 2.4 SVM CV Error vs C — Communities and Crime
# ------------------------------------------------------------------------------

png("stl_Ellora_HirudayaKumar_Le_svm_cv_crime.png", width = 800, height = 600)
plot(log10(svm_tune_c$performances$cost),
     svm_tune_c$performances$error,
     type = "b", pch = 19, col = "darkorange",
     xlab = "log10(C)", ylab = "10-fold CV MSE",
     main = "Communities and Crime SVM: CV Error vs Regularisation Parameter C")
abline(v = log10(best_C_c), col = "red", lty = 2)
legend("topright", legend = paste("Best C =", best_C_c), col = "red", lty = 2)
dev.off()
cat("Saved: stl_Ellora_HirudayaKumar_Le_svm_cv_crime.png\n")

# ------------------------------------------------------------------------------
# 2.5 Test MSE Comparison Bar Charts
# ------------------------------------------------------------------------------

png("stl_Ellora_HirudayaKumar_Le_mse_comparison.png", width = 1000, height = 700)
par(mfrow = c(1, 2), mar = c(10, 4, 3, 1))

boston_r <- results[results$Dataset == "Boston", ]
barplot(boston_r$Test_MSE, names.arg = boston_r$Method,
        las = 2, col = "steelblue", cex.names = 0.65,
        main = "Boston: Test MSE by Method", ylab = "Test MSE")

crime_r <- results[results$Dataset == "Communities and Crime", ]
barplot(crime_r$Test_MSE, names.arg = crime_r$Method,
        las = 2, col = "darkorange", cex.names = 0.65,
        main = "Communities and Crime: Test MSE by Method", ylab = "Test MSE")
dev.off()
cat("Saved: stl_Ellora_HirudayaKumar_Le_mse_comparison.png\n")

# ------------------------------------------------------------------------------
# 2.6 Execution Time Comparison
# ------------------------------------------------------------------------------

png("stl_Ellora_HirudayaKumar_Le_execution_time.png", width = 1000, height = 700)
par(mfrow = c(1, 2), mar = c(10, 4, 3, 1))

barplot(boston_r$Execution_Time_Sec, names.arg = boston_r$Method,
        las = 2, col = "steelblue", cex.names = 0.65,
        main = "Boston: Training Time by Method", ylab = "Time (seconds)")

barplot(crime_r$Execution_Time_Sec, names.arg = crime_r$Method,
        las = 2, col = "darkorange", cex.names = 0.65,
        main = "Communities and Crime: Training Time by Method",
        ylab = "Time (seconds)")
dev.off()
cat("Saved: stl_Ellora_HirudayaKumar_Le_execution_time.png\n")

cat("\nAll plots and summary table saved to working directory.\n")
cat("Script complete.\n")

# ==============================================================================
# END OF PLOTS SCRIPT
# ==============================================================================
