from dataclasses import dataclass
from itertools import product
from typing import Optional


@dataclass
class PrimitiveSet:
    bar_v: float = 70.0
    delta_v: float = 30.0
    delta_h: float = 0.95
    beta_H: float = 0.8
    beta_L: float = 0.98
    gamma: float = 0.20
    P0: float = 20.0
    Pm: float = 0.25
    rho: float = 10.0
    p_L: float = 0.50
    p_Y: float = 0.50
    firm_cost: float = 0.0
    n_hospitals: int = 200
    max_steps: int = 50
    seed: Optional[int] = 42
    delay_bias_H: float = 1.25
    delay_bias_L: float = 0.95

    def perceived_value(self, hospital_type: str) -> float:
        return self.bar_v + (self.delta_v if hospital_type == "L" else 0.0)

    @property
    def ratio_delta_v_bar_v(self) -> float:
        return self.delta_v / self.bar_v

    def contract_profit(self, contract: str, true_value: float) -> float:
        if contract == "Y":
            revenue = self.P0 + self.Pm * true_value
        elif contract == "S":
            revenue = self.gamma * true_value
        else:
            raise ValueError("Unknown contract")
        return revenue - self.firm_cost


def build_grid(
    bar_v_values=(50, 70, 100),
    beta_H_values=(0.7, 0.8, 0.9),
    gamma_values=(0.20, 0.50),
    P0_values=(10, 20, 30),
    p_L_values=(0.3, 0.5, 0.7),
    p_Y_values=(0.2, 0.5, 0.8),
    delta_v_values=(30,),
    delay_bias_H_values=(1.1, 1.25),
    delay_bias_L_values=(0.95, 1.0),
    n_hospitals=200,
    max_steps=50,
    seed=42,
):
    grid = []
    for vals in product(
        bar_v_values,
        beta_H_values,
        gamma_values,
        P0_values,
        p_L_values,
        p_Y_values,
        delta_v_values,
        delay_bias_H_values,
        delay_bias_L_values,
    ):
        bar_v, beta_H, gamma, P0, p_L, p_Y, delta_v, delay_bias_H, delay_bias_L = vals
        grid.append(
            PrimitiveSet(
                bar_v=bar_v,
                delta_v=delta_v,
                beta_H=beta_H,
                gamma=gamma,
                P0=P0,
                p_L=p_L,
                p_Y=p_Y,
                n_hospitals=n_hospitals,
                max_steps=max_steps,
                seed=seed,
                delay_bias_H=delay_bias_H,
                delay_bias_L=delay_bias_L,
            )
        )
    return grid
