import pandas as pd


def create_table(df,path):
    df_ot = df[df["Model"].str.contains("ot")]

    # W/o finetune
    df_no_fintune = df_ot[~df_ot["Model"].str.contains("finetune")]
    df_no_fintune['Model'] = df_no_fintune['Model'].str.slice(0, -12)

    # Finetune
    df_finetune = df_ot[df_ot["Model"].str.contains("40finetuned")]
    df_finetune['Model'] = df_finetune['Model'].str.slice(0, -12)

    # Group by cost
    df_feature_lp = df_no_fintune[df_no_fintune["Model"].str.contains("feature_lp")]
    df_quadratic = df_no_fintune[df_no_fintune["Model"].str.contains("quadratic_energy")]
    df_fused = df_no_fintune[df_no_fintune["Model"].str.contains("fused_gw")]
    df_fused_emd = df_fused[df_fused["Model"].str.contains("emd")]
    df_fused_emd = df_fused_emd[df_fused_emd["Model"].str.contains("2samples")]
    df_fused = df_fused[df_fused["Model"].str.contains("sinkhorn")]

    # Group the DataFrame by 'Model', calculate the mean, and get the index of the minimal mean
    df_feature_lp_grouped = df_feature_lp.groupby('Model').mean()
    df_feature_lp_grouped["std"] = df_feature_lp.groupby('Model').std()
    lp_min_mean_index = df_feature_lp_grouped[' MAE'].idxmin()
    df_lp_max_row = df_feature_lp_grouped.loc[lp_min_mean_index]

    df_quadratic_grouped = df_quadratic.groupby('Model').mean()
    df_quadratic_grouped["std"] = df_quadratic.groupby('Model').std()
    quadratic_min_mean_index = df_quadratic_grouped[' MAE'].idxmin()
    df_quadratic_max_row = df_quadratic_grouped.loc[quadratic_min_mean_index]

    df_fused_grouped = df_fused.groupby('Model').mean()
    df_fused_grouped["std"] = df_fused.groupby('Model').std()
    fused_min_mean_index = df_fused_grouped[' MAE'].idxmin()
    df_fused_max_row = df_fused_grouped.loc[fused_min_mean_index]

    # Create a new DataFrame
    df_final = pd.DataFrame({
        'model': [lp_min_mean_index, quadratic_min_mean_index,fused_min_mean_index],
        'mean': [df_lp_max_row[' MAE'], df_quadratic_max_row[' MAE'], df_fused_max_row[' MAE']],
        'std': [df_lp_max_row['std'], df_quadratic_max_row['std'], df_fused_max_row['std']]
    })

    ## Finetune
    df_feature_lp_finetune = df_finetune[df_finetune["Model"].str.contains(lp_min_mean_index)]
    df_quadratic_finetune = df_finetune[df_finetune["Model"].str.contains(quadratic_min_mean_index)]
    df_fused_finetune = df_finetune[df_finetune["Model"].str.contains(fused_min_mean_index)]


    finetune_rows = pd.DataFrame({
        'model': [lp_min_mean_index + "_finetune", quadratic_min_mean_index + "_finetune",fused_min_mean_index + "_finetune"],
        'mean': [df_feature_lp_finetune[" MAE"].mean(),df_quadratic_finetune[" MAE"].mean(),df_fused_finetune[" MAE"].mean()],
        'std': [df_feature_lp_finetune[" MAE"].std(),df_quadratic_finetune[" MAE"].std(),df_fused_finetune[" MAE"].std()]
    })

    df_final = df_final.append(finetune_rows, ignore_index=True)
    df_final.to_csv(path, index=False)

def main():

    df = pd.read_csv("results/optimization_MAE_emd.csv")
    path = "results/optimization_MAE_emd_table.csv"
    create_table(df,path)
    



if __name__ == '__main__':
    main()