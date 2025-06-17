from scipy.stats import ttest_rel

# 成對樣本資料
gemma = [0.433, 0.429, 0.583, 0.332, 0.366]
gemma_rag = [0.996, 0.695, 0.948, 0.939, 0.966]

llama = [0.464, 0.444, 0.659, 0.387, 0.416]
llama_rag = [1.000, 0.696, 0.989, 0.874, 1.000]

# 顯著水準
alpha = 0.1

def paired_t_test(data1, data2, model_name):
    t_stat, p_value = ttest_rel(data1, data2)
    print(f"\n模型：{model_name}")
    print(f"t 值: {t_stat:.4f}")
    print(f"p 值: {p_value:.4f}")
    if p_value < alpha:
        print(f"➡ 結果達顯著水準 α = {alpha}")
    else:
        print(f"➡ 結果未達顯著水準 α = {alpha}")

# 執行分析
paired_t_test(gemma, gemma_rag, "gemma:7b")
paired_t_test(llama, llama_rag, "llama3:8b")