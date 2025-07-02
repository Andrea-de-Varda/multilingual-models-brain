import statsmodels.api as sm

def forward_selection(data, target, predictors):
    initial_predictors = []  # Start with an empty model
    best_aic = float('inf')
    best_model = None

    while True:
        remaining_predictors = list(set(predictors) - set(initial_predictors))
        aic_candidates = []

        for predictor in remaining_predictors:
            current_predictors = initial_predictors + [predictor]
            X = sm.add_constant(data[current_predictors])
            model = sm.OLS(data[target], X).fit()
            aic_candidates.append((model.aic, predictor, model))

        # Find the predictor with the lowest AIC
        aic_candidates.sort()
        if aic_candidates and aic_candidates[0][0] < best_aic:
            best_aic, best_predictor, best_model = aic_candidates[0]
            initial_predictors.append(best_predictor)
        else:
            break

    return best_model, initial_predictors

def backward_elimination(data, target, predictors):
    best_aic = float('inf')
    best_model = None
    current_predictors = predictors.copy()

    while current_predictors:
        aic_candidates = []

        for predictor in current_predictors:
            remaining_predictors = list(set(current_predictors) - {predictor})
            X = sm.add_constant(data[remaining_predictors])
            model = sm.OLS(data[target], X).fit()
            aic_candidates.append((model.aic, predictor, model))

        # Find the predictor whose removal results in the lowest AIC
        aic_candidates.sort()
        if aic_candidates and aic_candidates[0][0] < best_aic:
            best_aic, removed_predictor, best_model = aic_candidates[0]
            current_predictors.remove(removed_predictor)
        else:
            break

    return best_model, current_predictors

def exhaustive_search(data, target, predictors):
    best_aic = float('inf')
    best_model = None
    best_combination = None

    for k in range(1, len(predictors) + 1):
        for combo in combinations(predictors, k):
            X = sm.add_constant(data[list(combo)])
            model = sm.OLS(data[target], X).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_combination = combo

    return best_model, best_combination


def calculate_aicc(model, n):
    aic = model.aic  # Extract AIC from the fitted model
    k = model.df_model + 1  # Number of parameters (including intercept)
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else float('inf')  # Avoid undefined AICc
    return aicc

def exhaustive_search_aicc(data, target, predictors):
    best_aicc = float('inf')
    best_model = None
    best_combination = None
    n = len(data)  # Sample size

    for k in range(1, len(predictors) + 1):
        for combo in combinations(predictors, k):
            X = sm.add_constant(data[list(combo)])
            model = sm.OLS(data[target], X).fit()
            aicc = calculate_aicc(model, n)
            if aicc < best_aicc:
                best_aicc = aicc
                best_model = model
                best_combination = combo

    return best_model, best_combination

predictors_multi = ["n_languages", "n_params", "ppx_mean", "mrr_multi"]
target_multi = "Score_multi"
best_model_multi, best_predictors_multi = exhaustive_search_aicc(all_data, target_multi, predictors_multi)
print("Best Predictors:", best_predictors_multi)
print(best_model_multi.summary())

predictors_mono = ["n_languages", "n_params", "ppx_mean", "mrr_mono"]
target_mono = "Score_mono"
best_model_mono, best_predictors_mono = exhaustive_search_aicc(all_data, target_mono, predictors_mono)
print("Best Predictors:", best_predictors_mono)
print(best_model_mono.summary())


#############################
# variance inflation factor #
#############################

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Define predictors
X_within = all_data[["n_languages", "n_params", "ppx_mean", "mrr_mono"]]
X_within = sm.add_constant(X_within)

vif_data = pd.DataFrame()
vif_data["Variable"] = X_within.columns
vif_data["VIF"] = [variance_inflation_factor(X_within.values, i) for i in range(X_within.shape[1])]

print(vif_data)

["nllb200_distilled_600M", "nllb200_distilled_1B", "nllb200_1B", "xlm_align", "infoxlm_base", "infoxlm_large", "multiminilm", "xlmr_base", "xlmr_large", "distilmbert", "bert_base", "mdeberta", "mt5_small", "mt5_base", "mt5_large", "mgpt","xglm_small", "xglm_med", "xglm_large", "xglm_xl"]

model_languages = [["nllb200_distilled_600M", 200,  615],
                   ["nllb200_distilled_1B",   200, 1371],
                   ["nllb200_1B",             200, 1371],
                   ["xlm_align",               94,  278],
                   ["infoxlm_base",            94,  278], 
                   ["infoxlm_large",           94,  560],
                   ["multiminilm",            100,  118],
                   ["xlmr_base",              100,  278],
                   ["xlmr_large",             100,  560],
                   ["distilmbert",            104,  135],
                   ["bert_base",              104,  178],
                   ["mdeberta",               100,  278],
                   ["mt5_small",              101,  172],
                   ["mt5_base",               101,  390],
                   ["mt5_large",              101,  973],
                   ["mgpt",                    60, 1418],
                   ["xglm_small",              30,  564],
                   ["xglm_med",                30, 1733],
                   ["xglm_large",              30, 2942],
                   ["xglm_xl",                 30, 4552]]

