library(lme4)
library(lmerTest)
library(dplyr)

setwd("/home/dev/Documents/PhD/Alice/other/synonyms")

############
# synonyms #
############

df_mono  <- read.csv("singlelangs_monolingual_results.csv")
df_multi <- read.csv("singlelangs_multilingual_results.csv")

m_mono  <- lmer("r ~ mrr + (mrr|model) + (mrr|language)", data=df_mono)
summary(m_mono)

m_multi  <- lmer("r ~ mrr + (1|model) + (mrr|language)", data=df_multi)
summary(m_multi)

# compare models with AIC
aic_mono <- AIC(m_mono)
aic_multi <- AIC(m_multi)
delta_aic <- abs(aic_mono - aic_multi)
evidence_ratio <- exp(delta_aic / 2)
cat("Delta AIC:", delta_aic, "\n")
cat("Evidence Ratio:", evidence_ratio, "\n")

##############
# perplexity #
##############

df_perplexity <- read.csv("perplexity_results.csv")
df_perplexity <- df_perplexity %>%
  rename(language = lang, model = mod)

merged_mono  <- merge(df_mono, df_perplexity, by = c("model", "language"))
merged_multi <- merge(df_multi, df_perplexity, by = c("model", "language"))

m_mono_p  <- lmer("r ~ ppx_residuals + (ppx_residuals|model) + (ppx_residuals|language)", data=merged_mono)
summary(m_mono_p)

m_multi_p  <- lmer("r ~ ppx_residuals + (ppx_residuals|model) + (ppx_residuals|language)", data=merged_multi)
summary(m_multi_p)

cor.test(merged_mono$r, merged_mono$ppx_residuals)
cor.test(merged_multi$r, merged_multi$ppx_residuals)

aic_mono_p <- AIC(m_mono_p)
aic_multi_p <- AIC(m_multi_p)
delta_aic_p <- abs(aic_mono_p - aic_multi_p)
evidence_ratio_p <- exp(delta_aic_p / 2)
cat("Delta AIC:", delta_aic_p, "\n")
cat("Evidence Ratio:", evidence_ratio_p, "\n")

###################################
# add also n° params and n° langs #
###################################

model_languages <- data.frame(
  model = c("nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", 
            "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", 
            "xlmr_base", "xlmr_large", "distilmbert", "bert_base", 
            "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt", 
            "xglm_small", "xglm_med", "xglm_large", "xglm_xl"),
  n_languages = c(200, 200, 200, 94, 94, 94, 100, 100, 100, 104, 104, 
                  100, 101, 101, 101, 60, 30, 30, 30, 30),
  n_params = c(615, 1371, 1371, 278, 278, 560, 118, 278, 560, 135, 178, 
               278, 172, 390, 973, 1418, 564, 1733, 2942, 4552)
)

merged_mono_all <- merge(
  merged_mono, 
  model_languages, 
  by = "model",
  all.x = TRUE)

# standardize continuous predictors
merged_mono_all$n_languages_z <- scale(merged_mono_all$n_languages)
merged_mono_all$n_params_z <- scale(merged_mono_all$n_params)
merged_mono_all$ppx_residuals_z <- scale(merged_mono_all$ppx_residuals)
merged_mono_all$mrr_z <- scale(merged_mono_all$mrr)

m_mono_p  <- lmer("r ~ n_languages_z + n_params_z + ppx_residuals_z + mrr_z + (1|model) + (mrr_z + ppx_residuals_z|language)", data=merged_mono_all)
summary(m_mono_p)

# multi
merged_multi_all <- merge(
  merged_multi, 
  model_languages, 
  by = "model",
  all.x = TRUE)

merged_multi_all$n_languages_z <- scale(merged_multi_all$n_languages)
merged_multi_all$n_params_z <- scale(merged_multi_all$n_params)
merged_multi_all$ppx_residuals_z <- scale(merged_multi_all$ppx_residuals)
merged_multi_all$mrr_z <- scale(merged_multi_all$mrr)

m_multi_p  <- lmer("r ~ n_languages_z + n_params_z + ppx_residuals_z + mrr_z + (1|model) + (mrr_z + ppx_residuals_z|language)", data=merged_multi_all)
summary(m_multi_p)
