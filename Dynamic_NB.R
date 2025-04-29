# =============================================================================
# Dynamic Naïve Bayes Filter for Bleeding Prediction
# =============================================================================

# --- ライブラリ読み込み ---
library(dplyr)
library(ggplot2)
library(pROC)
library(caret)
library(e1071)

# --- パラメータ ---
small <- 1e-6  # ゼロ分散対策用

# --- データ読み込み＋ラベル付与 ---
NB_bleeding <- read.csv("NB_bleeding_demo.csv", header = TRUE, sep = ",") %>%
  arrange(id, day) %>%
  group_by(id) %>%
  # 次の time の status をそのまま status_next として持ってくる
  mutate(status_next = lead(status)) %>%
  ungroup()

# --- 変数指定（静的・動的） ---
static_vars  <- names(NB_bleeding)[which(names(NB_bleeding) == "age"):
                                     which(names(NB_bleeding) == "main_disease")]
dynamic_vars <- names(NB_bleeding)[which(names(NB_bleeding) == "wbc"):
                                     which(names(NB_bleeding) == "gtp")]
all_vars <- c(static_vars, dynamic_vars)

# =============================================================================
# 条件付き分布の統計量推定
# =============================================================================
get_conditional_stats <- function(df, vars) {
  stats <- list()
  for (v in vars) {
    grp <- df %>%
      filter(!is.na(.data[[v]])) %>%
      group_by(status_next) %>%
      summarise(
        mean = mean(.data[[v]], na.rm = TRUE),
        sd   = sd(.data[[v]],   na.rm = TRUE),
        .groups = "drop"
      )
    # 両クラスの情報があるかチェック（なければ NA で埋める）
    mean1 <- grp$mean[grp$status_next == 1]
    sd1   <- grp$sd[grp$status_next == 1]
    mean0 <- grp$mean[grp$status_next == 0]
    sd0   <- grp$sd[grp$status_next == 0]
    stats[[v]] <- list(
      mean1 = ifelse(length(mean1)>0, mean1, NA),
      sd1   = ifelse(length(sd1  )>0, sd1,   NA),
      mean0 = ifelse(length(mean0)>0, mean0, NA),
      sd0   = ifelse(length(sd0  )>0, sd0,   NA)
    )
  }
  return(stats)
}

# =============================================================================
# 各変数の log-likelihood ratio を計算
# =============================================================================
compute_log_likelihood_ratio <- function(row, cond_stats) {
  log_ratio <- 0
  for (v in names(cond_stats)) {
    if (!is.na(cond_stats[[v]]$mean1) && !is.na(cond_stats[[v]]$mean0)) {
      x   <- as.numeric(row[[v]])
      if (!is.na(x)) {
        mu1 <- cond_stats[[v]]$mean1
        sd1 <- max(cond_stats[[v]]$sd1, small)
        mu0 <- cond_stats[[v]]$mean0
        sd0 <- max(cond_stats[[v]]$sd0, small)
        log_p1 <- dnorm(x, mean = mu1, sd = sd1, log = TRUE)
        log_p0 <- dnorm(x, mean = mu0, sd = sd0, log = TRUE)
        log_ratio <- log_ratio + (log_p1 - log_p0)
      }
    }
  }
  return(log_ratio)
}

# =============================================================================
# ログオッズを累積更新するフィルター(参考、これは使わない)
# =============================================================================
apply_log_bayes_filter <- function(data, cond_stats, prior_init = 0.5) {
  data <- data %>% arrange(id, day)
  data <- data %>% group_by(id) %>% mutate(
    score_dyn = {
      logit_prev <- log(prior_init / (1 - prior_init))
      n_pts <- n()
      scores <- numeric(n_pts)
      for (i in seq_len(n_pts)) {
        row_i <- cur_data()[i, ]
        delta  <- compute_log_likelihood_ratio(row_i, cond_stats)
        logit_now <- logit_prev + delta
        scores[i]  <- logit_now
        logit_prev <- logit_now
      }
      scores
    },
    prob_dyn = 1 / (1 + exp(-score_dyn))
  ) %>% ungroup()
  return(data)
}

# =============================================================================
# Dynamic Naïve Bayes Filter
# =============================================================================
apply_bayes_filter_with_transition <- function(data, cond_stats,
                                               P11 = 0.9, P01 = 0.1,
                                               prior_init = 0.5) {
  data <- data %>% arrange(id, day)
  data %>% group_by(id) %>% mutate(
    score_dyn = {
      # step0: 初期の logit
      logit_prev <- log(prior_init / (1 - prior_init))
      scores <- numeric(n())
      for (i in seq_len(n())) {
        # 1) 直前の posterior を確率に戻す
        p_prev <- 1 / (1 + exp(-logit_prev))
        # 2) 状態遷移で今回の事前確率を計算
        prior_t <- p_prev * P11 + (1 - p_prev) * P01
        logit_trans <- log(prior_t / (1 - prior_t))
        # 3) 観測尤度差（log-likelihood ratio）を計算
        row_i <- cur_data()[i, ]
        delta  <- compute_log_likelihood_ratio(row_i, cond_stats)
        # 4) 遷移後の logit に尤度差を足して更新
        logit_now <- logit_trans + delta
        scores[i] <- logit_now
        logit_prev <- logit_now
      }
      scores
    },
    prob_dyn = 1 / (1 + exp(-score_dyn))
  ) %>% ungroup()
}

# 学習／テストデータの分割
# =============================================================================
set.seed(123)
ids        <- unique(NB_bleeding$id)
train_ids  <- sample(ids, length(ids) * 0.7)
test_ids   <- setdiff(ids, train_ids)

train_data <- NB_bleeding %>% filter(id %in% train_ids, day %in% 0:5)
test_data  <- NB_bleeding %>% filter(id %in% test_ids,  day %in% 0:5)

# =============================================================================
# モデル学習と予測
# =============================================================================
cond_stats <- get_conditional_stats(train_data, all_vars)
test_results <- apply_bayes_filter_with_transition(test_data, cond_stats)

# =============================================================================
# リスク推移の可視化（ランダムに10 ID を抽出）
# =============================================================================
set.seed(42)
sample_ids <- sample(unique(test_results$id), 10, replace = FALSE)
ggplot(test_results %>% filter(id %in% sample_ids),
       aes(x = day, y = prob_dyn, group = id, color = as.factor(id))) +
  geom_line() +
  labs(
    title = "Bayes Filter NB (log space): Bleeding Risk Over Time",
    x = "Day", y = "Predicted Risk", color = "ID"
  ) +
  theme_minimal() +
  scale_color_viridis_d()

# =============================================================================
# AUROC（日別）の計算とプロット
# =============================================================================
auc_by_day <- test_results %>%
  group_by(day) %>%
  summarise(
    AUROC = tryCatch({
      roc_obj <- roc(status_next, prob_dyn)
      as.numeric(auc(roc_obj))
    }, error = function(e) NA),
    .groups = "drop"
  )

print(auc_by_day)

ggplot(auc_by_day, aes(x = day, y = AUROC)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Bayes Filter NB (log space): AUC Over Time",
    x = "Day", y = "AUROC"
  ) +
  theme_minimal()

# =============================================================================
# 混同行列（しきい値 0.5 で day=0～4）
# =============================================================================
threshold <- 0.5
for (d in 0:4) {
  cat("\n==== Confusion Matrix for Day", d, "====\n")
  df_day <- test_results %>% filter(day == d)
  if (nrow(df_day) > 0) {
    pred   <- factor(ifelse(df_day$prob_dyn >= threshold, 1, 0), levels = c(0,1))
    actual <- factor(df_day$status_next,                     levels = c(0,1))
    cm     <- confusionMatrix(pred, actual, positive = "1")
    print(cm$table)
    cat("Accuracy   :", round(cm$overall["Accuracy"], 3), "\n")
    cat("Sensitivity:", round(cm$byClass["Sensitivity"], 3), "\n")
    cat("Specificity:", round(cm$byClass["Specificity"], 3), "\n")
  } else {
    cat("No data for day", d, "\n")
  }
}

# =============================================================================
# 予測確率の分布確認
# =============================================================================
print(summary(test_results$prob_dyn))
hist(test_results$prob_dyn,
     breaks = 30,
     main   = "Histogram of Predicted Probabilities",
     xlab   = "prob_dyn")


# --- 1) Normal NB の AUROC（訓練データ上） ---
nb_auc_train <- data.frame(day = integer(), AUROC = numeric())

for (t in 0:5) {
  df_train <- train_data %>%
    filter(day == t) %>%
    select(all_of(all_vars), status_next) %>%
    na.omit()
  
  if (nrow(df_train) > 10 && length(unique(df_train$status_next)) > 1) {
    df_train$status_next <- factor(df_train$status_next, levels = c(0, 1))
    model <- naiveBayes(status_next ~ ., data = df_train)
    pred_prob <- predict(model, df_train, type = "raw")[, "1"]
    roc_obj <- roc(df_train$status_next, pred_prob, levels = c("0", "1"))
    auc_val <- as.numeric(auc(roc_obj))
  } else {
    auc_val <- NA
  }
  
  nb_auc_train <- rbind(nb_auc_train, data.frame(day = t, AUROC = auc_val))
}
nb_auc_train$model <- "Normal NB (train)"

# --- 2) Normal NB の AUROC（テストデータ上） ---
nb_auc_test <- data.frame(day = integer(), AUROC = numeric())

for (t in 0:5) {
  df_train <- train_data %>%
    filter(day == t) %>%
    select(all_of(all_vars), status_next) %>%
    na.omit()
  df_test  <- test_data %>%
    filter(day == t) %>%
    select(all_of(all_vars), status_next) %>%
    na.omit()
  
  if (nrow(df_train) > 10 && length(unique(df_train$status_next)) > 1 &&
      nrow(df_test)  > 10 && length(unique(df_test$status_next))  > 1) {
    df_train$status_next <- factor(df_train$status_next, levels = c(0, 1))
    model <- naiveBayes(status_next ~ ., data = df_train)
    pred_prob <- predict(model, df_test, type = "raw")[, "1"]
    roc_obj <- roc(df_test$status_next, pred_prob, levels = c("0", "1"))
    auc_val <- as.numeric(auc(roc_obj))
  } else {
    auc_val <- NA
  }
  
  nb_auc_test <- rbind(nb_auc_test, data.frame(day = t, AUROC = auc_val))
}
nb_auc_test$model <- "Normal NB (test)"

# --- 3) プロット比較 ---
auc_compare <- bind_rows(nb_auc_test, filter_auc)

ggplot(auc_compare, aes(x = day, y = AUROC, color = model)) +
  geom_line() +
  geom_point() +
  labs(title = "AUROC Over Time: Normal NB vs Bayes Filter NB",
       x = "Day", y = "AUROC",
       color = "Model") +
  theme_minimal()

# --- 4) 混同行列 (Normal NB on test data) の修正版 ---
for (t in 0:5) {
  cat("\n==== Confusion Matrix for Normal NB (Day", t, ") ====\n")
  
  df_train <- train_data %>%
    filter(day == t) %>%
    select(all_of(all_vars), status_next) %>%
    na.omit() %>%
    mutate(status_next = factor(status_next, levels = c(0, 1)))
  
  df_test <- test_data %>%
    filter(day == t) %>%
    select(all_of(all_vars), status_next) %>%
    na.omit() %>%
    mutate(status_next = factor(status_next, levels = c(0, 1)))  # ← ここで因子化
  
  if (nrow(df_train) > 10 && nrow(df_test) > 10 &&
      length(unique(df_train$status_next)) > 1 &&
      length(unique(df_test$status_next))  > 1) {
    
    # モデル学習
    model     <- naiveBayes(status_next ~ ., data = df_train)
    pred_prob <- predict(model, df_test, type = "raw")[, "1"]
    
    # 予測クラスも同じレベルで因子化
    pred_class <- factor(ifelse(pred_prob >= threshold, 1, 0),
                         levels = c(0, 1))
    
    # 混同行列
    cm <- confusionMatrix(pred_class, df_test$status_next, positive = "1")
    print(cm$table)
    cat("Accuracy:   ", round(cm$overall["Accuracy"], 3), "\n")
    cat("Sensitivity:", round(cm$byClass["Sensitivity"], 3), "\n")
    cat("Specificity:", round(cm$byClass["Specificity"], 3), "\n")
  } else {
    cat("Not enough data for day", t, "\n")
  }
}

