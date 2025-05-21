# --- ライブラリ読み込み ---
library(dplyr)
library(ggplot2)
library(pROC)
library(caret)
library(e1071)



# --- データ読み込み＋ラベル付与 ---
NB_bleeding <- read.csv("NB_bleeding_demo.csv", header = TRUE, sep = ",") %>%
  arrange(id, day) %>%
  group_by(id) %>%
  # 次の time の status をそのまま status_next として持ってくる
  mutate(status_next = lead(status)) %>%
  ungroup()

# 変数定義
static_vars  <- names(NB_bleeding)[which(names(NB_bleeding) == "age") : which(names(NB_bleeding) == "main_disease")]
dynamic_vars <- names(NB_bleeding)[which(names(NB_bleeding) == "wbc") : which(names(NB_bleeding) == "gtp")]
all_vars     <- c(static_vars, dynamic_vars)

# Cox用データ：idごとに1行にまとめる（動的変数は day == 0 の値を使用）
cox_data <- NB_bleeding %>%
  group_by(id) %>%
  summarise(
    time = max(time, na.rm = TRUE),
    status = max(status, na.rm = TRUE),
    across(all_of(static_vars), ~ first(.x), .names = "{.col}"),
    across(all_of(dynamic_vars), ~ .x[which.min(abs(day))], .names = "{.col}"),
    .groups = "drop"
  )

cox_formula <- as.formula(paste0("Surv(time, status) ~ ", paste(setdiff(names(cox_data), c("id", "time", "status")), collapse = " + ")))
cox_model <- coxph(cox_formula, data = cox_data)

# リスクスコア（線形予測子）を算出
cox_data$cox_risk <- predict(cox_model, type = "lp")

# AUROC（全体で1点）
cox_auc <- roc(cox_data$status, cox_data$cox_risk)
auc_cox <- as.numeric(auc(cox_auc))

nb_day0 <- test_results_raw %>%
  filter(day == 0) %>%
  filter(!is.na(status_next))

nb_auc_day0 <- roc(nb_day0$status_next, nb_day0$prob_dyn)
auc_nb <- as.numeric(auc(nb_auc_day0))
# データフレーム作成
auc_df <- data.frame(
  model = c("Cox Model", "Naive Bayes"),
  AUROC = c(auc_cox, auc_nb)
)

# プロット
ggplot(auc_df, aes(x = model, y = AUROC, fill = model)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = round(AUROC, 3)), vjust = -0.5) +
  ylim(0, 1) +
  theme_minimal() +
  labs(title = "AUROC Comparison: Cox vs Naive Bayes",
       x = NULL, y = "AUROC")

library(broom)
library(ggplot2)

cox_tidy <- tidy(cox_model, exponentiate = TRUE, conf.int = TRUE)

# termの順番をHR順に整える（見栄え）
cox_tidy <- cox_tidy %>%
  filter(term != "(Intercept)") %>%
  mutate(term = reorder(term, estimate))

# フォレストプロット
ggplot(cox_tidy, aes(x = estimate, y = term)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray40") +
  scale_x_log10() +
  labs(
    title = "Adjusted Hazard Ratios (Cox Model)",
    x = "Hazard Ratio (log scale)", y = NULL
  ) +
  theme_minimal()

library(survival)
library(survminer)  # 生存曲線のggplotラッパー

# KM推定
km_fit <- survfit(Surv(time, status) ~ anticoag_hepa, data = cox_data)

# プロット
ggsurvplot(
  km_fit,
  data = cox_data,
  risk.table = TRUE,           # 下にリスクテーブル表示
  pval = TRUE,                 # log-rank検定のp値
  conf.int = TRUE,             # 信頼区間の帯を表示
  legend.labs = c("0", "1"),
  xlab = "Time (days)",
  ylab = "Survival Probability",
  title = "Kaplan-Meier Survival Curve by Heparin Usage",
  palette = "Dark2"
)

set.seed(123)
ids <- unique(cox_data$id)
train_ids <- sample(ids, length(ids) * 0.7)
test_ids  <- setdiff(ids, train_ids)

cox_train <- cox_data %>% filter(id %in% train_ids)
cox_test  <- cox_data %>% filter(id %in% test_ids)

# モデル学習
cox_model <- coxph(Surv(time, status) ~ ..., data = cox_train)

# 線形予測子（リスクスコア）をテストデータに適用
cox_test$cox_risk <- predict(cox_model, newdata = cox_test, type = "lp")

# テストデータ上でAUROC算出
cox_auc <- roc(cox_test$status, cox_test$cox_risk)
auc_cox <- as.numeric(auc(cox_auc))

auc_cox