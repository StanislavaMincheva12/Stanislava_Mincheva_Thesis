import random
import pandas as pd
from mesa import Model
from mesa.time import BaseScheduler

from Agents.Firm import Firm
from Agents.Hospital import Hospital


class AdoptionModel(Model):
    def __init__(self, primitives):
        super().__init__()
        self.primitives = primitives
        self.random = random.Random(primitives.seed)
        self.schedule = BaseScheduler(self)
        self.firm = Firm(self)
        self.posterior_p_L = primitives.p_L
        self.adoptions = []
        self.posterior_history = []

        for i in range(primitives.n_hospitals):
            hospital_type = "L" if self.random.random() < primitives.p_L else "H"
            h = Hospital(self, hospital_type=hospital_type, unique_id=i)
            self.schedule.add(h)

    def register_adoption(self, hospital):
        self.adoptions.append(
            {
                "id": hospital.unique_id,
                "type": hospital.hospital_type,
                "delay": hospital.delay_count,
                "adoption_time": hospital.adoption_time,
                "contract": hospital.accepted_contract,
            }
        )

    def update_posterior(self):
        active_agents = [a for a in self.schedule.agents if a.state == "active"]
        if not active_agents:
            return
        delayed = [a for a in active_agents if a.delay_count > 0]
        if delayed:
            observed_share_L = sum(a.hospital_type == "L" for a in delayed) / len(delayed)
            self.posterior_p_L = 0.5 * self.posterior_p_L + 0.5 * observed_share_L
        self.posterior_history.append({"step": self.schedule.steps, "posterior_p_L": self.posterior_p_L})

    def step(self):
        self.firm.choose_contract()
        self.schedule.step()
        self.update_posterior()

    def run_model(self):
        while self.schedule.steps < self.primitives.max_steps:
            active_agents = [a for a in self.schedule.agents if a.state == "active"]
            if not active_agents:
                break
            self.step()
        return pd.DataFrame(self.adoptions), pd.DataFrame(self.firm.history), pd.DataFrame(self.posterior_history)
