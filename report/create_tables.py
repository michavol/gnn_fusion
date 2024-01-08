import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")


def create_table(df, path, sample_size_filter_lp_q=None, sample_size_filter_gw=None, algo_filter=None):
    df_ot = df[df["Model"].str.contains("ot")]
    # Filter
    df_ot = df_ot[~df_ot["Model"].str.contains("acts_True")]
    df_ot = df_ot[~df_ot["Model"].str.contains("acts_[True]")]
    if algo_filter is not None:
        df_ot = df_ot[df_ot["Model"].str.contains(algo_filter)]

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

    # Group the DataFrame by 'Model', calculate the mean, and get the index of the minimal mean
    if sample_size_filter_lp_q is not None:
        df_feature_lp = df_feature_lp[df_feature_lp["Model"].str.contains(f"{sample_size_filter_lp_q}samples")]
    df_feature_lp_grouped = df_feature_lp.groupby('Model').mean()
    df_feature_lp_grouped["std"] = df_feature_lp.groupby('Model').std()
    lp_min_mean_index = df_feature_lp_grouped[' MAE'].idxmin()
    df_lp_max_row = df_feature_lp_grouped.loc[lp_min_mean_index]

    if sample_size_filter_lp_q is not None:
        df_quadratic = df_quadratic[df_quadratic["Model"].str.contains(f"{sample_size_filter_lp_q}samples")]
    df_quadratic_grouped = df_quadratic.groupby('Model').mean()
    df_quadratic_grouped["std"] = df_quadratic.groupby('Model').std()
    quadratic_min_mean_index = df_quadratic_grouped[' MAE'].idxmin()
    df_quadratic_max_row = df_quadratic_grouped.loc[quadratic_min_mean_index]

    if sample_size_filter_gw is not None:
        df_fused = df_fused[df_fused["Model"].str.contains(f"{sample_size_filter_gw}samples")]
    df_fused_grouped = df_fused.groupby('Model').mean()
    df_fused_grouped["std"] = df_fused.groupby('Model').std()
    fused_min_mean_index = df_fused_grouped[' MAE'].idxmin()
    df_fused_max_row = df_fused_grouped.loc[fused_min_mean_index]

    # Create a new DataFrame
    df_final = pd.DataFrame({
        'model': [lp_min_mean_index, quadratic_min_mean_index, fused_min_mean_index],
        'mean': [df_lp_max_row[' MAE'], df_quadratic_max_row[' MAE'], df_fused_max_row[' MAE']],
        'std': [df_lp_max_row['std'], df_quadratic_max_row['std'], df_fused_max_row['std']]
    })

    ## Finetune
    df_feature_lp_finetune = df_finetune[df_finetune["Model"].str.contains(lp_min_mean_index)]
    df_quadratic_finetune = df_finetune[df_finetune["Model"].str.contains(quadratic_min_mean_index)]
    df_fused_finetune = df_finetune[df_finetune["Model"].str.contains(fused_min_mean_index)]

    finetune_rows = pd.DataFrame({
        'model': [lp_min_mean_index + "_finetune", quadratic_min_mean_index + "_finetune",
                  fused_min_mean_index + "_finetune"],
        'mean': [df_feature_lp_finetune[" MAE"].mean(), df_quadratic_finetune[" MAE"].mean(),
                 df_fused_finetune[" MAE"].mean()],
        'std': [df_feature_lp_finetune[" MAE"].std(), df_quadratic_finetune[" MAE"].std(),
                df_fused_finetune[" MAE"].std()]
    })

    # df_final = df_final.append(finetune_rows, ignore_index=True)
    df_final = pd.concat([df_final, pd.DataFrame(finetune_rows)], ignore_index=True)
    df_final.to_csv(path, index=False)

    for index, row in df_final.iterrows():
        print(f"{row['model']}: {row['mean']}+/-{row['std']}")

def main():
    parser = argparse.ArgumentParser(description='Filtering the results.')

    parser.add_argument('table_name', type=str, help='Path to the result table.')
    parser.add_argument('--algo', type=str, help='OT algo.', default=None)
    parser.add_argument('--sample_size_lp_q', type=int, default=None)
    parser.add_argument('--sample_size_gw', type=int, default=None)

    args = parser.parse_args()

    # table_name = "results/optimization_MAE_emd.csv"
    df = pd.read_csv(args.table_name)
    path = args.table_name.split('.')[0] + "_results.csv"
    create_table(df, path, sample_size_filter_lp_q=args.sample_size_lp_q,
                 sample_size_filter_gw=args.sample_size_gw, algo_filter=args.algo)


if __name__ == '__main__':
    main()
