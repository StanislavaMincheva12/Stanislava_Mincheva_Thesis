class Firm:
    def __init__(self, model):
        self.model = model
        self.current_contract = None
        self.switch_time = None
        self.history = []

    def choose_contract(self):
        p = self.model.posterior_p_L
        true_v_L = self.model.primitives.bar_v + self.model.primitives.delta_v
        true_v_H = self.model.primitives.bar_v

        profit_Y = (
            p * self.model.primitives.contract_profit("Y", true_v_L)
            + (1 - p) * self.model.primitives.contract_profit("Y", true_v_H)
        )
        profit_S = (
            p * self.model.primitives.contract_profit("S", true_v_L)
            + (1 - p) * self.model.primitives.contract_profit("S", true_v_H)
        )

        contract = "Y" if profit_Y >= profit_S else "S"
        if self.current_contract == "Y" and contract == "S" and self.switch_time is None:
            self.switch_time = self.model.schedule.steps
        self.current_contract = contract
        self.history.append(
            {
                "step": self.model.schedule.steps,
                "posterior_p_L": p,
                "profit_Y": profit_Y,
                "profit_S": profit_S,
                "contract": contract,
            }
        )
        return contract
