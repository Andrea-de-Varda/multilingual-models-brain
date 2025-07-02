# r analysis
library(lme4)
library(lmerTest)
library(sjPlot)
library(sjlabelled)
library(sjmisc)
library(ggplot2)
library(dplyr)

data <-read.csv("../results/transfer_analysis_R.csv")
colnames(data)

# standardizing predictors
data_standardized <- data
predictors_to_standardize <- c("syn", "geo", "pho", "gen", "inv", "feat", "ppx_1", "ppx_2")
for(predictor in predictors_to_standardize) {
  data_standardized[[predictor]] <- scale(data[[predictor]])
}

# random intercepts
m0 <- lmer(transfer ~ syn + geo + pho + gen + inv + feat + ppx_1 + ppx_2 + (1|model) + (1|Lang1) + (1|Lang2), data = data_standardized)
summary(m0)

# crossed random effects
data_expanded <- rbind(
  transform(data_standardized, Language = Lang1),
  transform(data_standardized, Language = Lang2)
)


m0_crossed <- lmer(transfer ~ syn + geo + pho + gen + inv + feat + ppx_1 + ppx_2 + (1|model) + (1|Language), data = data_expanded)
summary(m0_crossed)

m0_crossed <- lmer(transfer ~ syn + ppx_1 + ppx_2 + (1|model) + (1|Language), data = data_expanded)
summary(m0_crossed)
m0_crossed <- lmer(transfer ~ geo + ppx_1 + ppx_2 + (1|model) + (1|Language), data = data_expanded)
summary(m0_crossed)
m0_crossed <- lmer(transfer ~ pho + ppx_1 + ppx_2 + (1|model) + (1|Language), data = data_expanded)
summary(m0_crossed)
m0_crossed <- lmer(transfer ~ gen + ppx_1 + ppx_2 + (1|model) + (1|Language), data = data_expanded)
summary(m0_crossed)
m0_crossed <- lmer(transfer ~ inv + ppx_1 + ppx_2 + (1|model) + (1|Language), data = data_expanded)
summary(m0_crossed)
m0_crossed <- lmer(transfer ~ feat + ppx_1 + ppx_2 + (1|model) + (1|Language), data = data_expanded)
summary(m0_crossed)



# perplexity, L1
data_singlelang <-read.csv("../results/perplexity_results.csv")
colnames(data_singlelang)

data_standardized_singlelang <- data_singlelang
predictors_to_standardize <- c("ppx", "ppx_random", "r", "m", "part_corr")
for(predictor in predictors_to_standardize) {
  data_standardized_singlelang[[predictor]] <- scale(data_singlelang[[predictor]])
}

m0_ppx <- lmer(m ~ ppx + ppx_random + part_corr + (1|mod), data = data_standardized_singlelang)
summary(m0_ppx)

m1_ppx <- lmer(r ~ m + ppx + ppx_random + part_corr + (1|mod), data = data_standardized_singlelang)
summary(m1_ppx)


p <- plot_model(m0_ppx, type = "std", show.values = TRUE, value.offset = .3, dot.size = 3,
                axis.labels = c("Reliability", "Perplexity (random)", "Perplexity"),
                width = .5,
                title = " ")

p + theme_light(base_size = 20) +
  theme(panel.border = element_rect(color = "black", fill = NA)) +
  theme(panel.background = element_rect(fill = "white"),
        axis.text = element_text(color = "black"),        # Set axis text color to black
        axis.title = element_text(color = "black"),       # Set axis title color to black
        axis.ticks = element_line(color = "black")) +     # Set axis ticks color to black
  labs(y = "β") +
  geom_hline(yintercept = 0, linetype = 'dotted', col = 'black', size = 1) 


p <- plot_model(m1_ppx, type = "std", show.values = TRUE, value.offset = .3, dot.size = 3,
                axis.labels = c("Reliability", "Perplexity (random)", "Perplexity", "Monolingual encoding"),
                width = .5,
                title = " ")

p + theme_light(base_size = 20) +
  theme(panel.border = element_rect(color = "black", fill = NA)) +
  theme(panel.background = element_rect(fill = "white"),
        axis.text = element_text(color = "black"),        # Set axis text color to black
        axis.title = element_text(color = "black"),       # Set axis title color to black
        axis.ticks = element_line(color = "black")) +     # Set axis ticks color to black
  labs(y = "β") +
  geom_hline(yintercept = 0, linetype = 'dotted', col = 'black', size = 1) 

