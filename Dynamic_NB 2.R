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

NB_bleeding <- NB_bleeding %>%
  select(status_next, everything())

NB_bleeding$status_next[!is.na(NB_bleeding$status_next)] <- 0
NB_bleeding$status[!is.na(NB_bleeding$status_next)] <- NB_bleeding$status_next[!is.na(NB_bleeding$status_next)]

NB_bleeding <- NB_bleeding %>%
  group_by(id) %>% mutate(status_next = lead(status)) %>%
  ungroup()

NB_bleeding <- NB_bleeding %>%
  filter(!is.na(status_next))

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
      filter(!is.na(.data[[v]]), !is.na(status_next)) %>% #修正しました
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
apply_log_bayes_filter_weighted <- function(data, cond_stats, prior_init = 0.5, alpha = 0.5) {
  data <- data %>% arrange(id, day)
  data <- data %>% group_by(id) %>% mutate(
    score_logadd = {
      logit_prev <- log(prior_init / (1 - prior_init))
      n_pts <- n()
      scores <- numeric(n_pts)
      for (i in seq_len(n_pts)) {
        row_i <- cur_data()[i, ]
        delta  <- compute_log_likelihood_ratio(row_i, cond_stats)
        # 重み付き更新（過去と現在を alpha : 1-alpha で混合）
        # 原案はalphaなしのlogit_now <- logit_prev + delta
        logit_now <- alpha * logit_prev + (1 - alpha) * delta
        scores[i]  <- logit_now
        logit_prev <- logit_now
      }
      scores
    },
    prob_logadd = 1 / (1 + exp(-score_logadd))
  ) %>% ungroup()
  return(data)
}
#_dynをlogaddに修正
#alphaを加えた。alpha = 1だと前回分、alpha = 0だと今回分のみの観測尤度

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

# =============================================================================
#' Dynamic Naive Bayes Filter with personalized transition probabilities
#' (P11 and P01 computed from static features at each time point)
# =============================================================================

apply_bayes_filter_with_transition_custom <- function(data, cond_stats,
                                                      beta = c(-2, 3, 0.05),
                                                      prior_init = 0.5) {
  # beta: coefficients for logistic model to compute P(Y_t = 1 | Y_{t-1}, static_vars)
  # beta[1] = intercept, beta[2] = coefficient for Y_{t-1}, beta[3:] = static feature coeffs
  
  data <- data %>% arrange(id, day)
  
  # 必要な静的変数の列を抽出（ここでは age だけを例示）
  static_vars <- c("age")  # 必要に応じて拡張
  
  data %>% group_by(id) %>% mutate(
    score_dyn = {
      logit_prev <- log(prior_init / (1 - prior_init))
      scores <- numeric(n())
      
      # static変数を1人ぶんだけ取り出す（id単位）
      static_vals <- cur_data()[1, static_vars, drop = TRUE] %>% as.numeric()
      
      for (i in seq_len(n())) {
        # 1) 前回の事後を確率に戻す
        p_prev <- 1 / (1 + exp(-logit_prev))
        
        # 2) 状態遷移確率をロジスティックで計算（個別化）
        y_prev <- round(p_prev)  # 簡易的に 0/1 として扱う
        z <- beta[1] + beta[2] * y_prev + sum(beta[3:length(beta)] * static_vals)
        prior_t <- 1 / (1 + exp(-z))
        
        # 3) 観測尤度差を計算
        row_i <- cur_data()[i, ]
        delta <- compute_log_likelihood_ratio(row_i, cond_stats)
        
        # 4) ロジット更新
        logit_now <- log(prior_t / (1 - prior_t)) + delta
        scores[i] <- logit_now
        logit_prev <- logit_now
      }
      scores
    },
    prob_dyn = 1 / (1 + exp(-score_dyn))
  ) %>% ungroup()
}

# =============================================================================
#' Dynamic Naive Bayes Filter with personalized transition probabilities, random forest
#' (P11 and P01 computed from static features at each time point)
# =============================================================================

library(randomForest)

apply_bayes_filter_with_transition_rf <- function(data, cond_stats, rf_model, prior_init = 0.5) {
  data <- data %>% arrange(id, day)
  
  # 静的変数の名前（id列など除く）
  static_vars <- setdiff(names(rf_model$forest$xlevels), "y_prev")
  
  data <- data %>% group_by(id) %>% mutate(
    score_dyn = {
      logit_prev <- log(prior_init / (1 - prior_init))
      scores <- numeric(n())
      
      # idごとの静的変数は1回だけ抽出
      static_vals <- cur_data()[1, static_vars, drop = FALSE]
      
      for (i in seq_len(n())) {
        # 前回の事後確率 → y_prev = 0 or 1 に変換（離散化）
        y_prev <- as.integer(round(1 / (1 + exp(-logit_prev))))
        
        # 状態遷移確率の予測
        row_rf <- cbind(y_prev = y_prev, static_vals)
        prior_t <- predict(rf_model, newdata = row_rf, type = "prob")[, "1"]
        
        # 観測尤度差（動的変数を用いて計算）
        row_i <- cur_data()[i, ]
        delta <- compute_log_likelihood_ratio(row_i, cond_stats)
        
        # ベイズ更新
        logit_trans <- log(prior_t / (1 - prior_t))
        logit_now <- logit_trans + delta
        scores[i] <- logit_now
        logit_prev <- logit_now
      }
      
      scores
    },
    prob_dyn = 1 / (1 + exp(-score_dyn))
  ) %>% ungroup()
  
  return(data)
}

# =============================================================================
# PCA
# =============================================================================

apply_bayes_filter_with_transition_pca <- function(data, cond_stats,
                                                   P11 = 0.9, P01 = 0.1,
                                                   prior_init = 0.5) {
  data <- data %>% arrange(id, day)
  
  # PCAデータでlog-likelihood ratioを計算 → 逐次的に更新
  data <- data %>% group_by(id) %>% mutate(
    score_dyn = {
      logit_prev <- log(prior_init / (1 - prior_init))
      scores <- numeric(n())
      
      for (i in seq_len(n())) {
        p_prev <- 1 / (1 + exp(-logit_prev))
        prior_t <- p_prev * P11 + (1 - p_prev) * P01
        logit_trans <- log(prior_t / (1 - prior_t))
        
        # PCA成分でlog-likelihoodを計算
        row_i <- cur_data()[i, ]
        delta <- compute_log_likelihood_ratio(row_i, cond_stats)
        
        logit_now <- logit_trans + delta
        scores[i] <- logit_now
        logit_prev <- logit_now
      }
      
      scores
    },
    prob_dyn = 1 / (1 + exp(-score_dyn))
  ) %>% ungroup()
  
  return(data)
}

# 学習／テストデータの分割
# =============================================================================
set.seed(123)
ids        <- unique(NB_bleeding$id)
train_ids  <- sample(ids, length(ids) * 0.7)
test_ids   <- setdiff(ids, train_ids)

train_data <- NB_bleeding %>% filter(id %in% train_ids, day %in% 0:5)
test_data  <- NB_bleeding %>% filter(id %in% test_ids,  day %in% 0:5)

cond_stats_raw <- get_conditional_stats(train_data, all_vars)
test_results_raw <- apply_bayes_filter_with_transition(test_data, cond_stats_raw)

auc_raw <- test_results_raw %>%
  group_by(day) %>%
  summarise(
    AUROC = tryCatch({ as.numeric(auc(roc(status_next, prob_dyn))) }, error = function(e) NA),
    .groups = "drop"
  ) %>%
  mutate(model = "No PCA")

pca_dims <- c(1, 2, 3)
auc_pca_list <- list()

# PCA前に欠損除去したX_trainを使う
X_train <- train_data %>%
  select(all_of(all_vars), status_next) %>%
  na.omit()

pca_model <- prcomp(X_train[, all_vars], center = TRUE, scale. = TRUE)

X_test <- test_data %>%
  select(all_of(all_vars), status_next, id, day) %>%
  na.omit()

for (k in pca_dims) {
  # PCA変換
  train_proj <- predict(pca_model, X_train[, all_vars])[, 1:k, drop = FALSE]
  train_proj_df <- as.data.frame(train_proj)
  train_proj_df$status_next <- X_train$status_next
  
  test_proj <- predict(pca_model, X_test[, all_vars])[, 1:k, drop = FALSE]
  test_proj_df <- as.data.frame(test_proj)
  test_proj_df$status_next <- X_test$status_next
  test_proj_df$id <- X_test$id
  test_proj_df$day <- X_test$day
  
  # モデル学習・予測
  vars <- colnames(train_proj_df)[1:k]
  cond_stats <- get_conditional_stats(train_proj_df, vars)
  test_results <- apply_bayes_filter_with_transition_pca(test_proj_df, cond_stats)
  
  # AUROC計算
  auc_df <- test_results %>%
    filter(!is.na(status_next)) %>%
    group_by(day) %>%
    summarise(
      AUROC = if (n_distinct(status_next) >= 2) {
        tryCatch({
          as.numeric(auc(roc(status_next, prob_dyn)))
        }, error = function(e) NA)
      } else {
        NA
      },
      .groups = "drop"
    ) %>%
    mutate(model = paste0("PCA (", k, "D)"))
  
  auc_pca_list[[paste0("PCA_", k, "D")]] <- auc_df
}

auc_compare <- bind_rows(auc_raw, do.call(bind_rows, auc_pca_list))

ggplot(auc_compare, aes(x = day, y = AUROC, color = model)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Bayes Filter NB: AUROC Over Time (PCA vs No PCA)",
    x = "Day", y = "AUROC", color = "Model"
  ) +
  ylim(0, 1) +
  theme_minimal()

rf_train <- train_data %>%
  group_by(id) %>%
  mutate(y_prev = lag(status)) %>%
  filter(!is.na(status_next), !is.na(y_prev)) %>%
  ungroup()

static_formula <- paste(static_vars, collapse = " + ")
rf_formula <- as.formula(paste("as.factor(status_next) ~ y_prev +", static_formula))

rf_model <- randomForest(
  formula = rf_formula,
  data = rf_train,
  ntree = 100
)

# 条件付き分布の統計量（動的変数）を推定
cond_stats <- get_conditional_stats(train_data, dynamic_vars)

# フィルター適用
test_results_rf <- apply_bayes_filter_with_transition_rf(
  test_data, cond_stats, rf_model
)

library(pROC)

auc_by_day_rf <- test_results_rf %>%
  filter(!is.na(status_next)) %>%
  group_by(day) %>%
  summarise(
    AUROC = tryCatch({
      roc_obj <- roc(status_next, prob_dyn)
      as.numeric(auc(roc_obj))
    }, error = function(e) NA),
    .groups = "drop"
  ) %>%
  mutate(model = "RF-Bayes Filter")

auc_pca_1d <- auc_pca_list[["PCA_1D"]]

auc_pca_1d

# 既に作ってある Naive Bayes の AUROC 結果などと結合
auc_compare <- bind_rows(auc_by_day_rf, auc_raw, auc_pca_1d)

ggplot(auc_compare, aes(x = day, y = AUROC, color = model)) +
  geom_line(size = 0.5) +
  geom_point(size = 1) +
  labs(
    title = "Time-dependent AUROC: NB vs RF-Bayes vs PCA",
    x = "Day", y = "AUROC", color = "Model"
  ) +
  theme_minimal()




# =============================================================================
# モデル学習と予測
# =============================================================================
cond_stats <- get_conditional_stats(train_data, all_vars)
test_results <- apply_bayes_filter_with_transition_custom(test_data, cond_stats)

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
auc_compare <- bind_rows(nb_auc_test, auc_by_day)

ggplot(auc_compare, aes(x = day, y = AUROC, color = model)) +
  geom_line() +
  geom_point() +
  labs(title = "AUROC Over Time: Normal NB vs Bayes Filter NB Custom",
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
