import pandas as pd



def create_table_emd(df,path):

    #df_sinkhorn = df[df["Model"].str.contains("sinkhorn")]
    df_ot = df[df["Model"].str.contains("ot")]

    df_no_fintune = df_ot[~df_ot["Model"].str.contains("finetune")]

    # feature lp
    df_feature_lp = df_no_fintune[df_no_fintune["Model"].str.contains("feature_lp")]

    # quadratic
    df_quadratic_02 = df_no_fintune[df_no_fintune["Model"].str.contains("quadratic_energy_alpha_02")]
    df_quadratic_05 = df_no_fintune[df_no_fintune["Model"].str.contains("quadratic_energy_alpha_05")]
    df_quadratic_08 = df_no_fintune[df_no_fintune["Model"].str.contains("quadratic_energy_alpha_08")]

    # fused gw
    df_fused_primal = df_no_fintune[df_no_fintune["Model"].str.contains("fused_gw_large_eps_primal")]
    df_fused_reg = df_no_fintune[df_no_fintune["Model"].str.contains("fused_gw_large_eps_reg")]

    # compute mean and std
    df_feature_lp_mean = df_feature_lp[" MAE"].mean()
    df_feature_lp_std = df_feature_lp[" MAE"].std()
    
    df_quadratic_02_mean = df_quadratic_02[" MAE"].mean()
    df_quadratic_02_std = df_quadratic_02[" MAE"].std()
    df_quadratic_05_mean = df_quadratic_05[" MAE"].mean()
    df_quadratic_05_std = df_quadratic_05[" MAE"].std()
    df_quadratic_08_mean = df_quadratic_08[" MAE"].mean()
    df_quadratic_08_std = df_quadratic_08[" MAE"].std()

    df_qaudratic_mean = pd.DataFrame({
        'model': ['quadratic_02', 'quadratic_05', 'quadratic_08'],
        'mean': [df_quadratic_02_mean, df_quadratic_05_mean, df_quadratic_08_mean],
        'std': [df_quadratic_02_std, df_quadratic_05_std, df_quadratic_08_std]
    })
    
    df_quadratic_max_idx = df_qaudratic_mean['mean'].idxmin()
    df_quadratic_max_row = df_qaudratic_mean.loc[df_quadratic_max_idx]
    
    df_fused_primal_mean = df_fused_primal[" MAE"].mean()
    df_fused_primal_std = df_fused_primal[" MAE"].std()
    df_fused_reg_mean = df_fused_reg[" MAE"].mean()
    df_fused_reg_std = df_fused_reg[" MAE"].std()

    df_fused_mean = pd.DataFrame({
        'model': ['fused_primal', 'fused_reg'],
        'mean': [df_fused_primal_mean, df_fused_reg_mean],
        'std': [ df_fused_primal_std, df_fused_reg_std]
    })


    df_fused_max_idx = df_fused_mean['mean'].idxmin()
    df_fused_max_row = df_fused_mean.loc[df_fused_max_idx]


    # Create a new DataFrame
    df_final = pd.DataFrame({
        'model': ['feature_lp',  df_quadratic_max_row['model'], df_fused_max_row['model']],
        'mean': [df_feature_lp_mean, df_quadratic_max_row['mean'], df_fused_max_row['mean']],
        'std': [df_feature_lp_std, df_quadratic_max_row['std'], df_fused_max_row['std']]
    })

    df_final.to_csv(path, index=False)

def create_table_sinkhorn(df,path):
    df_ot = df[df["Model"].str.contains("ot")]

    df_no_fintune = df_ot[~df_ot["Model"].str.contains("finetune")]
    # Remove the last 5 letters from every string in the 'model' column
    df_no_fintune['Model'] = df_no_fintune['Model'].str.slice(0, -12)

    #print(df_no_fintune["Model"].iloc[1])

    df_feature_lp = df_no_fintune[df_no_fintune["Model"].str.contains("feature_lp")]
    df_quadratic = df_no_fintune[df_no_fintune["Model"].str.contains("quadratic_energy")]
    df_fused = df_no_fintune[df_no_fintune["Model"].str.contains("fused_gw_large")]

    # Group the DataFrame by 'Model', calculate the mean, and get the index of the minimal mean
    df_feature_lp_grouped = df_feature_lp.groupby('Model').mean()
    df_feature_lp_grouped["std"] = df_feature_lp.groupby('Model').std()
    lp_min_mean_index = df_feature_lp_grouped[' MAE'].idxmin()
    df_lp_max_row = df_feature_lp_grouped.loc[lp_min_mean_index]

    print(df_feature_lp_grouped)

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

    df_final.to_csv(path, index=False)

def main():

    # df = pd.read_csv("results/optimization_MAE_emd.csv")
    # path = "results/optimization_MAE_emd_table.csv"
    # create_table_sinkhorn(df,path)
    
    # df_unbalanced = pd.read_csv("results/optimization_MAE_unbalanced.csv")
    # path_unbalanced = "results/optimization_MAE_emd_unbalanced_table.csv"
    # create_table_emd(df_unbalanced,path_unbalanced)

    df_sinkhorn = pd.read_csv("results/optimization_MAE_sinkhorn.csv")
    path_sinkhorn = "results/optimization_MAE_skinhorn_table.csv"
    create_table_sinkhorn(df_sinkhorn,path_sinkhorn)



if __name__ == '__main__':
    main()