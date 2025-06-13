suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(e1071)
  library(pROC)
  library(caret)
  library(ggplot2)
  library(tibble)
  library(randomForest)
})

# 1. データの読み込みと並べ替え
NB_bleeding <- read.csv("NB_bleeding_demo.csv") %>%
  arrange(id, day)

# 2. 正しい「日ごとイベント」と「目的変数」の作成
NB_bleeding_processed <- NB_bleeding %>%
  filter(day <= 6) %>%
  group_by(id) %>%
  mutate(
    # ルールに基づき「daily_event」（日ごとイベント）を作成
    daily_event = if_else(status == 1 & day == max(day), 1, 0),
    # この daily_event を元に、モデルに必要な変数を作成
    status_next = lead(daily_event),
    y_prev = lag(daily_event)
  ) %>%
  ungroup() %>%
  filter(!is.na(status_next)) # 最後の行は status_next がNAになるため除外

# --- 変数指定（静的・動的） ---
static_vars  <- names(NB_bleeding)[which(names(NB_bleeding) == "age"):
                                     which(names(NB_bleeding) == "main_disease")]
dynamic_vars <- names(NB_bleeding)[which(names(NB_bleeding) == "wbc"):
                                     which(names(NB_bleeding) == "gtp")]
all_vars <- c(static_vars, dynamic_vars)
small        <- 1e-6

## ── 2. ヘルパー関数群 ───────────────────────────────────────────────────────
get_conditional_stats <- function(df, vars){
  map(vars, function(v) {
    g <- df %>% filter(!is.na(.data[[v]])) %>%
      group_by(status_next) %>%
      summarise(mu = mean(.data[[v]]), sd = sd(.data[[v]]), .groups = "drop")
    list(
      mean1 = g$mu[g$status_next==1] %||% NA_real_,
      sd1   = g$sd[g$status_next==1] %||% NA_real_,
      mean0 = g$mu[g$status_next==0] %||% NA_real_,
      sd0   = g$sd[g$status_next==0] %||% NA_real_
    )
  }) %>% setNames(vars)
}

compute_log_likelihood_ratio <- function(row, stats){
  reduce(names(stats), .init = 0, function(acc, v) {
    s <- stats[[v]]; x <- row[[v]]
    if(!is.na(x) && !is.na(s$mean1) && !is.na(s$mean0)){
      sd1 <- max(s$sd1, small, na.rm = TRUE); sd0 <- max(s$sd0, small, na.rm = TRUE)
      acc + dnorm(x, s$mean1, sd1, log=TRUE) - dnorm(x, s$mean0, sd0, log=TRUE)
    } else {
      acc
    }
  })
}


safe_auc <- function(truth, prob){
  truth <- if(is.factor(truth)) as.numeric(as.character(truth)) else truth
  if(length(unique(truth)) < 2) return(NA_real_)
  tryCatch({
    as.numeric(pROC::auc(truth, prob, quiet = TRUE, levels = c(0,1), direction = "<"))
  }, error = function(e) NA_real_)
}

train_nb_daywise <- function(train_df, test_df, vars){
  train_df <- train_df %>% select(all_of(vars), status_next) %>% na.omit()
  test_df  <- test_df  %>% select(all_of(vars), status_next) %>% na.omit()
  if(nrow(train_df)<10 || length(unique(train_df$status_next))<2 || nrow(test_df)<2) return(NA_real_)
  mdl <- naiveBayes(status_next ~ ., data = train_df %>% mutate(status_next=factor(status_next)))
  preds <- predict(mdl, test_df, type="raw")[,"1"]
  safe_auc(test_df$status_next, preds)
}

train_rf_daywise <- function(train_df, test_df, vars){
  # 必要な変数を選択し、NAを含む行を削除
  train_df <- train_df %>% select(all_of(vars), status_next) %>% na.omit()
  test_df  <- test_df  %>% select(all_of(vars), status_next) %>% na.omit()
  
  # データが少ない場合や、片方のクラスしか存在しない場合はNAを返す
  if(nrow(train_df) < 10 || length(unique(train_df$status_next)) < 2 || nrow(test_df) < 2) {
    return(NA_real_)
  }
  
  # ランダムフォレストモデルを学習
  # 目的変数は因子型（factor）に変換する
  mdl <- randomForest(factor(status_next) ~ ., data = train_df, ntree = 100)
  
  # テストデータを予測し、AUCを計算
  preds <- predict(mdl, newdata = test_df, type = "prob")[, "1"]
  safe_auc(test_df$status_next, preds)
}


apply_full_personalized_filter <- function(data, rf_initial_model, rf_transition_model, cond_stats, static_vars, all_vars) {
  data %>% arrange(id, day) %>%
    group_by(id) %>%
    mutate(
      score_dyn_personalized = {
        n_days <- n()
        scores <- numeric(n_days)
        
        # 初期確率の予測には Day0 の全変数を使用
        day0_data_for_pred <- cur_data()[1, all_vars, drop = FALSE]
        # 遷移確率の予測には静的変数のみ使用
        patient_static_data <- cur_data()[1, static_vars, drop = FALSE]
        
        # ★★★ 初期確率の予測に使う newdata を変更 ★★★
        prob_init <- predict(rf_initial_model, newdata = day0_data_for_pred, type = "prob")[, "1"]
        logit_prev <- log(prob_init / (1 - prob_init))
        
        for (i in seq_len(n_days)) {
          # このループ内の遷移確率の予測は変更なし
          y_prev_factor <- factor(round(plogis(logit_prev)), levels = c("0", "1"))
          rf_trans_input <- cbind(y_prev = y_prev_factor, patient_static_data)
          prior_t <- predict(rf_transition_model, newdata = rf_trans_input, type = "prob")[, "1"]
          logit_trans <- log(prior_t / (1 - prior_t))
          
          delta <- compute_log_likelihood_ratio(cur_data()[i, ], cond_stats)
          print(delta)
          logit_now <- logit_trans + delta
          
          scores[i] <- logit_now
          logit_prev <- logit_now
        }
        scores
      },
      prob_dyn_personalized = plogis(score_dyn_personalized)
    ) %>%
    ungroup()
}


# DynNB_Personalized2用の関数（Day 0だけdeltaを足さないバージョン）
apply_dynnb_personalized2 <- function(data, rf_initial_model, rf_transition_model, cond_stats, static_vars, all_vars) {
  data %>% arrange(id, day) %>%
    group_by(id) %>%
    mutate(
      score_dyn_p2 = {
        n_days <- n()
        scores <- numeric(n_days)
        
        day0_data_for_pred <- cur_data()[1, all_vars, drop = FALSE]
        patient_static_data <- cur_data()[1, static_vars, drop = FALSE]
        
        prob_init <- predict(rf_initial_model, newdata = day0_data_for_pred, type = "prob")[, "1"]
        logit_prev <- log(prob_init / (1 - prob_init))
        
        for (i in seq_len(n_days)) {
          
          # --- 条件分岐 ---
          if (i == 1) { # Day 0 (最初のステップ) の場合
            # deltaを足さず、rf_initialの予測値をそのまま使う
            logit_now <- logit_prev 
          } else { # Day 1 以降の場合
            # 通常通り、遷移確率＋尤度のベイズ更新を行う
            y_prev_factor <- factor(round(plogis(logit_prev)), levels = c("0", "1"))
            rf_trans_input <- cbind(y_prev = y_prev_factor, patient_static_data)
            prior_t <- predict(rf_transition_model, newdata = rf_trans_input, type = "prob")[, "1"]
            logit_trans <- log(prior_t / (1 - prior_t))
            
            delta <- compute_log_likelihood_ratio(cur_data()[i, ], cond_stats)
            logit_now <- logit_trans + delta
          }
          
          scores[i] <- logit_now
          logit_prev <- logit_now
        }
        scores
      },
      prob_dyn_p2 = plogis(score_dyn_p2)
    ) %>%
    ungroup()
}

## ── 3. 5分割交差検証 (5-Fold CV) ──────────────────────────────────────────
set.seed(123)
k <- 5
folds <- createFolds(unique(NB_bleeding_processed$id), k = k)

# 結果を格納するリストを準備
res_dyn_p <- list()
res_dyn_p2 <- list() # ★ DynNB_Personalized2の結果を格納するリストを追加
res_nb <- list()
res_rf_dw <- list()

message(sprintf("Running %d-Fold Patient-Level Cross-Validation with 4 models...", k))

for(f in seq_along(folds)){
  message(sprintf("Processing Fold %d/%d...", f, k))
  
  test_ids  <- unique(NB_bleeding_processed$id)[folds[[f]]]
  train_ids <- setdiff(unique(NB_bleeding_processed$id), test_ids)
  
  train_data <- NB_bleeding_processed %>% filter(id %in% train_ids)
  test_data  <- NB_bleeding_processed %>% filter(id %in% test_ids)
  
  # --- 1. 動的モデル用のコンポーネント学習 ---
  # DynNB_Personalized と DynNB_Personalized2 は同じ学習済みモデルを共有する
  rf_initial <- NULL
  rf_transition <- NULL
  st <- NULL
  
  if(sum(train_data$status_next) > 0) {
    # (a) 初期状態予測モデル
    train_initial <- train_data %>% 
      filter(day == 0) %>% 
      select(all_of(all_vars), status_next)
    rf_initial <- randomForest(factor(status_next) ~ ., data = train_initial, ntree = 100)
    
    # (b) 状態遷移予測モデル
    train_transition <- train_data %>%
      filter(!is.na(y_prev)) %>%
      select(all_of(static_vars), y_prev, status_next)
    train_transition$y_prev <- factor(train_transition$y_prev, levels = c("0", "1"))
    rf_transition <- randomForest(factor(status_next) ~ ., data = train_transition, ntree = 100)
    
    # (c) 観測尤度用の統計量
    st <- get_conditional_stats(train_data, dynamic_vars)
  }
  
  # --- 2. 各モデルの予測と評価 ---
  
  # 2a. DynNB_Personalized
  if (!is.null(rf_initial)) {
    pred_dyn_p <- apply_full_personalized_filter(
      data = test_data, rf_initial_model = rf_initial, rf_transition_model = rf_transition,
      cond_stats = st, static_vars = static_vars, all_vars = all_vars
    )
    auc_dyn_p <- pred_dyn_p %>% group_by(day) %>%
      summarise(AUROC = safe_auc(status_next, prob_dyn_personalized), .groups="drop") %>%
      mutate(fold = f, model = "DynNB_Personalized")
    res_dyn_p[[f]] <- auc_dyn_p
  }
  
  # ★ 2b. DynNB_Personalized2 (Day0特別扱い版)
  if (!is.null(rf_initial)) {
    pred_dyn_p2 <- apply_dynnb_personalized2(
      data = test_data, rf_initial_model = rf_initial, rf_transition_model = rf_transition,
      cond_stats = st, static_vars = static_vars, all_vars = all_vars
    )
    auc_dyn_p2 <- pred_dyn_p2 %>% group_by(day) %>%
      summarise(AUROC = safe_auc(status_next, prob_dyn_p2), .groups="drop") %>%
      mutate(fold = f, model = "DynNB_Personalized2")
    res_dyn_p2[[f]] <- auc_dyn_p2
  }
  
  # 2c. NB_daywise
  auc_nb <- map_dfr(unique(train_data$day), function(d) {
    train_d <- train_data %>% filter(day == d); test_d  <- test_data  %>% filter(day == d)
    tibble(day = d, AUROC = train_nb_daywise(train_d, test_d, all_vars))
  }) %>% mutate(fold = f, model = "NB_daywise")
  res_nb[[f]] <- auc_nb
  
  # 2d. RF_daywise
  auc_rf_dw <- map_dfr(unique(train_data$day), function(d) {
    train_d <- train_data %>% filter(day == d); test_d  <- test_data  %>% filter(day == d)
    tibble(day = d, AUROC = train_rf_daywise(train_d, test_d, all_vars))
  }) %>% mutate(fold = f, model = "RF_daywise")
  res_rf_dw[[f]] <- auc_rf_dw
}

message("Cross-Validation finished.")

## ── 4. 結果の集計 ──────────────────────────────────────────────────────────
# ★ bind_rowsに新しいモデルの結果を追加
cv_results <- bind_rows(res_dyn_p, res_dyn_p2, res_nb, res_rf_dw)

agg_results <- cv_results %>%
  group_by(model, day) %>%
  summarise(
    mean_AUROC = mean(AUROC, na.rm = TRUE),
    sd_AUROC   = sd(AUROC, na.rm = TRUE),
    .groups = "drop"
  )

print("Aggregated AUROC Results:")
print(agg_results)

## ── 5. プロット ─────────────────────────────────────────────────────────────
ggplot(agg_results, aes(x = day, y = mean_AUROC, color = model, group = model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  geom_errorbar(
    aes(ymin = mean_AUROC - sd_AUROC, ymax = mean_AUROC + sd_AUROC),
    width = 0.2, alpha = 0.7
  ) +
  scale_y_continuous(limits = c(0.4, 1.0), breaks = seq(0.4, 1, 0.1)) +
  scale_x_continuous(breaks = unique(agg_results$day)) +
  labs(
    title = "5-Fold CV: Personalized Dynamic NB vs. Baseline NB",
    subtitle = "AUROC by Day",
    x = "Day",
    y = "Mean AUROC (±1 SD)",
    color = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )