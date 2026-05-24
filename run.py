from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from Primitives.Primitives import build_grid
from Model.Model import AdoptionModel


def summarize_run(primitives, adoption_df, firm_df):
    if adoption_df.empty:
        return {
            "bar_v": primitives.bar_v,
            "delta_v": primitives.delta_v,
            "delta_v_over_bar_v": primitives.ratio_delta_v_bar_v,
            "beta_H": primitives.beta_H,
            "beta_L": primitives.beta_L,
            "gamma": primitives.gamma,
            "P0": primitives.P0,
            "p_L": primitives.p_L,
            "p_Y": primitives.p_Y,
            "switch_time_Y_to_S": None,
            "avg_delay_H": 0.0,
            "avg_delay_L": 0.0,
            "n_L_accept_Y": 0,
            "n_L_accept_S": 0,
            "n_H_accept_Y": 0,
            "n_H_accept_S": 0,
            "all_adopted": False,
        }

    switch_time = None
    if not firm_df.empty:
        switch_mask = (firm_df["contract"].shift(1) == "Y") & (firm_df["contract"] == "S")
        if switch_mask.any():
            switch_time = firm_df.loc[switch_mask, "step"].min()

    return {
        "bar_v": primitives.bar_v,
        "delta_v": primitives.delta_v,
        "delta_v_over_bar_v": primitives.ratio_delta_v_bar_v,
        "beta_H": primitives.beta_H,
        "beta_L": primitives.beta_L,
        "gamma": primitives.gamma,
        "P0": primitives.P0,
        "p_L": primitives.p_L,
        "p_Y": primitives.p_Y,
        "switch_time_Y_to_S": switch_time,
        "avg_delay_H": adoption_df.loc[adoption_df["type"] == "H", "delay"].mean(),
        "avg_delay_L": adoption_df.loc[adoption_df["type"] == "L", "delay"].mean(),
        "n_L_accept_Y": ((adoption_df["type"] == "L") & (adoption_df["contract"] == "Y")).sum(),
        "n_L_accept_S": ((adoption_df["type"] == "L") & (adoption_df["contract"] == "S")).sum(),
        "n_H_accept_Y": ((adoption_df["type"] == "H") & (adoption_df["contract"] == "Y")).sum(),
        "n_H_accept_S": ((adoption_df["type"] == "H") & (adoption_df["contract"] == "S")).sum(),
        "all_adopted": len(adoption_df) == primitives.n_hospitals,
    }


def build_interpretation_table(summary_df):
    df = summary_df.copy()

    total_L = df["n_L_accept_Y"] + df["n_L_accept_S"]
    total_H = df["n_H_accept_Y"] + df["n_H_accept_S"]

    df["share_L_Y"] = (df["n_L_accept_Y"] / total_L).fillna(0.0)
    df["share_L_S"] = (df["n_L_accept_S"] / total_L).fillna(0.0)
    df["share_H_Y"] = (df["n_H_accept_Y"] / total_H).fillna(0.0)
    df["share_H_S"] = (df["n_H_accept_S"] / total_H).fillna(0.0)
    df["delay_gap_H_minus_L"] = df["avg_delay_H"].fillna(0.0) - df["avg_delay_L"].fillna(0.0)
    df["L_delays_less_than_H"] = df["delay_gap_H_minus_L"] > 0
    df["firm_switches"] = df["switch_time_Y_to_S"].notna()

    nice = df[[
        "bar_v", "delta_v_over_bar_v", "beta_H", "gamma", "P0", "p_L", "p_Y",
        "avg_delay_H", "avg_delay_L", "delay_gap_H_minus_L", "L_delays_less_than_H",
        "switch_time_Y_to_S", "firm_switches",
        "n_L_accept_Y", "n_L_accept_S", "share_L_Y", "share_L_S",
        "all_adopted"
    ]].copy()

    for col in ["avg_delay_H", "avg_delay_L", "delay_gap_H_minus_L", "share_L_Y", "share_L_S"]:
        nice[col] = nice[col].round(3)

    return nice


def make_visuals(nice_df, output_dir):
    vis_dir = output_dir / "visuals"
    vis_dir.mkdir(exist_ok=True)

    delay_df = nice_df.groupby("beta_H")[["avg_delay_H", "avg_delay_L"]].mean().reset_index()
    plt.figure(figsize=(7, 4))
    plt.plot(delay_df["beta_H"], delay_df["avg_delay_H"], marker="o", label="High-bias")
    plt.plot(delay_df["beta_H"], delay_df["avg_delay_L"], marker="o", label="Low-bias")
    plt.xlabel("beta_H")
    plt.ylabel("Average delay")
    plt.title("Average delay by hospital type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "delay_by_beta_H.png", dpi=200)
    plt.close()

    contract_df = nice_df.groupby(["gamma"])[["share_L_Y", "share_L_S"]].mean().reset_index()
    plt.figure(figsize=(7, 4))
    width = 0.18
    x = range(len(contract_df))
    plt.bar([i - width / 2 for i in x], contract_df["share_L_Y"], width=width, label="Low-bias share on c_Y")
    plt.bar([i + width / 2 for i in x], contract_df["share_L_S"], width=width, label="Low-bias share on c_S")
    plt.xticks(list(x), [str(v) for v in contract_df["gamma"]])
    plt.xlabel("gamma")
    plt.ylabel("Average share")
    plt.title("Low-bias contract acceptance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "low_bias_contract_shares.png", dpi=200)
    plt.close()

    switch_df = nice_df.groupby(["gamma", "P0"])["firm_switches"].mean().reset_index()
    pivot = switch_df.pivot(index="P0", columns="gamma", values="firm_switches")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Share of runs with switch")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel("gamma")
    plt.ylabel("P0")
    plt.title("Firm switch frequency")
    plt.tight_layout()
    plt.savefig(vis_dir / "firm_switch_frequency.png", dpi=200)
    plt.close()


def print_key_answers(nice_df):
    print("\n=== INTERPRETABLE SUMMARY TABLE ===")
    print(nice_df.to_string(index=False))

    print("\n=== DIRECT ANSWERS TO THESIS QUESTIONS ===")
    print("1) High-bias hospital delay varies with primitives through avg_delay_H.")
    print(nice_df[["bar_v", "delta_v_over_bar_v", "beta_H", "gamma", "P0", "p_L", "p_Y", "avg_delay_H"]].to_string(index=False))

    print("\n2) Low-bias hospitals delay less than high-bias hospitals when L_delays_less_than_H = True.")
    print(nice_df[["bar_v", "beta_H", "gamma", "P0", "p_L", "p_Y", "avg_delay_H", "avg_delay_L", "L_delays_less_than_H"]].to_string(index=False))

    print("\n3) The firm switches from c_Y to c_S when firm_switches = True; switch_time_Y_to_S gives the first switch period.")
    print(nice_df[["bar_v", "beta_H", "gamma", "P0", "p_L", "p_Y", "firm_switches", "switch_time_Y_to_S"]].to_string(index=False))

    print("\n4) Low-bias contract acceptance is given by n_L_accept_Y, n_L_accept_S and their shares.")
    print(nice_df[["bar_v", "beta_H", "gamma", "P0", "p_L", "p_Y", "n_L_accept_Y", "n_L_accept_S", "share_L_Y", "share_L_S"]].to_string(index=False))


def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    grid = build_grid()

    for i, primitives in enumerate(grid):
        model = AdoptionModel(primitives)
        adoption_df, firm_df, posterior_df = model.run_model()

        case_dir = output_dir / f"case_{i:03d}"
        case_dir.mkdir(exist_ok=True)
        adoption_df.to_csv(case_dir / "adoptions.csv", index=False)
        firm_df.to_csv(case_dir / "firm_history.csv", index=False)
        posterior_df.to_csv(case_dir / "posterior_history.csv", index=False)
        summaries.append(summarize_run(primitives, adoption_df, firm_df))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    nice_df = build_interpretation_table(summary_df)
    nice_df.to_csv(output_dir / "summary_interpretable.csv", index=False)
    make_visuals(nice_df, output_dir)
    print_key_answers(nice_df)
    print("\nSaved files:")
    print("- results/summary.csv")
    print("- results/summary_interpretable.csv")
    print("- results/visuals/delay_by_beta_H.png")
    print("- results/visuals/low_bias_contract_shares.png")
    print("- results/visuals/firm_switch_frequency.png")


if __name__ == "__main__":
    main()