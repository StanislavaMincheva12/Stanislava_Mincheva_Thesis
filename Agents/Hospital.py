from mesa import Agent


class Hospital(Agent):
    def __init__(self, model, hospital_type: str, unique_id: int):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.hospital_type = hospital_type
        self.state = "active"
        self.delay_count = 0
        self.adoption_time = None
        self.accepted_contract = None

    @property
    def beta(self):
        return self.model.primitives.beta_L if self.hospital_type == "L" else self.model.primitives.beta_H

    @property
    def perceived_value(self):
        return self.model.primitives.perceived_value(self.hospital_type)

    def utility_if_accept(self, contract: str):
        p = self.model.primitives
        if contract == "Y":
            payment = p.P0 + p.Pm * self.perceived_value
        elif contract == "S":
            payment = p.gamma * self.perceived_value
        else:
            raise ValueError("Unknown contract")
        return self.beta * p.delta_h * (self.perceived_value - payment) - p.rho

    def perceived_try_value(self):
        y_val = self.utility_if_accept("Y")
        s_val = self.utility_if_accept("S")
        return self.model.primitives.p_Y * max(y_val, 0.0) + (1 - self.model.primitives.p_Y) * max(s_val, 0.0)

    def perceived_delay_value(self):
        base = self.beta * self.model.primitives.delta_h * self.perceived_try_value()
        if self.hospital_type == "H":
            return base * self.model.primitives.delay_bias_H
        return base * self.model.primitives.delay_bias_L

    def step(self):
        if self.state != "active":
            return

        try_val = self.perceived_try_value()
        delay_val = self.perceived_delay_value()

        if delay_val > try_val and self.delay_count < self.model.primitives.max_steps:
            self.delay_count += 1
            return

        contract = self.model.firm.current_contract
        accept_utility = self.utility_if_accept(contract)
        if accept_utility >= 0:
            self.state = "adopted"
            self.adoption_time = self.model.schedule.steps
            self.accepted_contract = contract
            self.model.register_adoption(self)
        else:
            fallback = "S" if contract == "Y" else "Y"
            fallback_utility = self.utility_if_accept(fallback)
            if fallback_utility >= 0:
                self.state = "adopted"
                self.adoption_time = self.model.schedule.steps
                self.accepted_contract = fallback
                self.model.register_adoption(self)
            else:
                self.delay_count += 1
