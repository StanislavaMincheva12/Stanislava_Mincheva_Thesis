"""
Firm Agent Module
=================

This module implements the Firm (seller) agent in the firm-hospital game.

Theory Overview
---------------
The Firm is a strategic, unbiased seller with the following characteristics:
    - Discount factor: δ_f (exponential discounting)
    - Belief about hospital types: p_L ∈ (0,1)
    - Updates belief via Bayes' rule based on observed behavior
    - Chooses contracts to maximize expected profit
    
    p_L = Pr(η_L | observed behavior)
        = P(hospital is low-bias)
    
Firm Strategy
-------------
The firm observes the cohort behavior and updates its belief p_L.
Based on p_L, it decides which contract to offer via PURE STRATEGY:

Decision Rule (Threshold):
    If E[π(c_Y | p_L)] >= E[π(c_S | p_L)]:  Offer c_Y
    Else:                                    Offer c_S

Expected Profit Calculation
----------------------------
For a cohort where p_L is the believed fraction of low-bias hospitals:

π(c_Y | p_L) = p_L · (Revenue from low-bias) - C
             = p_L · [P(c_Y, v_actual) - C]

π(c_S | p_L) = E[Revenue from all hospitals] - C
             = [p_L · P(c_S, v_actual) + (1-p_L) · P(c_S, v̄)] - C

Note: High-bias hospitals reject c_Y, so firm only gets payment from
      low-bias hospitals for c_Y contracts.

Bayesian Belief Updating
------------------------
The firm observes which hospitals:
    - TRY (move to contract decision)
    - DELAY (will be reconsidered next period)
    - REJECT (will not adopt)

The firm infers:
    - More TRYs relative to total: suggests higher fraction of low-bias
    - More DELAYs: suggests higher fraction of high-bias (impatient)
    
The update uses Bayes' rule:
    p_L^(t+1) ∝ p_L^(t) · P(observed | p_L^(t))

Pure Strategy Commitment
------------------------
Unlike screening games, the firm commits to a pure strategy:
    - No randomization
    - Clear threshold rule
    - Deterministic given belief state
    
This is because:
    1. Hospitals don't learn from firm's choice (fixed p_Y belief)
    2. Firm has no incentive to hide information
    3. Pure strategy equilibrium aligns with analytical results
"""

import numpy as np
from enum import Enum
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FirmParameters:
    """
    Hyperparameters for firm agent.
    
    Attributes
    ----------
    delta_f : float
        Firm's exponential discount factor (δ_f ∈ (0, 1))
        Represents firm's impatience
    c_cost : float
        Fixed operational cost per contract (C > 0)
    """
    delta_f: float = 0.95
    c_cost: float = 15.0


class FirmContractChoice(Enum):
    """Enumeration of firm contract choices."""
    CONTRACT_Y = "c_Y"
    CONTRACT_S = "c_S"


class Firm:
    """
    Firm Agent
    ==========
    
    A strategic, unbiased seller in the firm-hospital game.
    
    The firm observes hospital behavior and updates its belief about
    the composition of the population (what fraction are low-bias).
    Based on this belief, it makes a pure strategy choice about which
    contract to offer.
    
    Attributes
    ----------
    agent_id : int
        Unique identifier for the firm
    params : FirmParameters
        Structural parameters
    p_L : float
        Firm's current belief about P(hospital is low-bias)
    p_L_history : List[float]
        History of belief updates over periods
    contract_history : List[str]
        History of contract choices
    payoff : float
        Cumulative payoff
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        agent_id: int,
        params: FirmParameters,
        initial_p_L: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize a Firm agent.
        
        Parameters
        ----------
        agent_id : int
            Unique identifier
        params : FirmParameters
            Game structural parameters
        initial_p_L : float, optional
            Initial belief about fraction of low-bias hospitals. Default 0.5.
        seed : int, optional
            Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.params = params
        self.p_L = initial_p_L
        self.rng = np.random.RandomState(seed)
        
        # History tracking
        self.p_L_history = [initial_p_L]
        self.contract_history = []
        self.payoff = 0.0
        self.revenue_from_cY = 0.0
        self.revenue_from_cS = 0.0
        self.num_cY_adopted = 0
        self.num_cS_adopted = 0
    
    # =========================================================================
    # BELIEF MANAGEMENT
    # =========================================================================
    
    @property
    def p_H(self) -> float:
        """
        Firm's belief about P(hospital is high-bias).
        
        Returns
        -------
        float
            1 - p_L
        """
        return 1 - self.p_L
    
    def update_belief_bayesian(
        self,
        num_tries: int,
        num_delays: int,
        num_rejects: int,
        p_try_given_L: float = 0.85,
        p_try_given_H: float = 0.35
    ) -> float:
        """
        Update belief p_L using Bayes' rule.
        
        Theory: Given observed actions in a cohort, update the belief
        about the fraction of low-bias hospitals using Bayes' rule.
        
        The firm observes:
            - Low-bias hospitals: tend to TRY (accept easily)
            - High-bias hospitals: tend to DELAY or REJECT (impatient)
        
        Likelihood: P(TRY | type) and P(DELAY | type)
        
        Parameters
        ----------
        num_tries : int
            Number of hospitals that chose TRY
        num_delays : int
            Number of hospitals that chose DELAY
        num_rejects : int
            Number of hospitals that chose REJECT
        p_try_given_L : float, optional
            P(TRY | η_L) = likelihood of trying given low-bias
        p_try_given_H : float, optional
            P(TRY | η_H) = likelihood of trying given high-bias
        
        Returns
        -------
        float
            Updated belief p_L
        """
        total = num_tries + num_delays + num_rejects
        
        if total == 0:
            return self.p_L
        
        # Likelihoods for observing this many TRYs from a population
        # Use binomial-like calculation
        p_delay_given_L = 0.10
        p_delay_given_H = 0.55
        p_reject_given_L = 0.05
        p_reject_given_H = 0.10
        
        # Probability of observed pattern given low-bias population
        likelihood_L = (
            (p_try_given_L ** num_tries) *
            (p_delay_given_L ** num_delays) *
            (p_reject_given_L ** num_rejects)
        )
        
        # Probability of observed pattern given high-bias population
        likelihood_H = (
            (p_try_given_H ** num_tries) *
            (p_delay_given_H ** num_delays) *
            (p_reject_given_H ** num_rejects)
        )
        
        # Bayesian update: P(L | obs) ∝ P(obs | L) · P(L)
        p_L_new = (likelihood_L * self.p_L) / (
            likelihood_L * self.p_L + likelihood_H * (1 - self.p_L) + 1e-10
        )
        
        # Bound to [0.01, 0.99] to avoid extremes
        p_L_new = np.clip(p_L_new, 0.01, 0.99)
        
        self.p_L = p_L_new
        self.p_L_history.append(p_L_new)
        
        return p_L_new
    
    # =========================================================================
    # PROFIT CALCULATIONS
    # =========================================================================
    
    def expected_profit_from_contract(
        self,
        contract: str,
        v_low_bias: float,
        v_high_bias: float,
        p_0: float,
        p_m_coeff: float,
        gamma: float,
        delta_h: float
    ) -> float:
        """
        Calculate expected profit from offering a contract given current belief p_L.
        
        Theory: Expected profit depends on:
            1. Probability hospital is low-bias (p_L)
            2. Which type accepts which contract
            3. Revenue if accepted
            4. Fixed cost C
        
        For c_Y (usage-based):
            - Only low-bias hospitals accept
            - Payment: P_0 + δ_h · P_m(v)
            - E[π(c_Y)] = p_L · [Payment - C]
        
        For c_S (subscription):
            - Both types accept
            - Payment: δ_h · γ · v
            - E[π(c_S)] = [p_L · Payment_L + (1-p_L) · Payment_H] - C
        
        Parameters
        ----------
        contract : str
            Contract type ('c_Y' or 'c_S')
        v_low_bias : float
            Low-bias hospital's perceived value
        v_high_bias : float
            High-bias hospital's perceived value
        p_0 : float
            Upfront fee for c_Y
        p_m_coeff : float
            Coefficient for usage-based fee
        gamma : float
            Subscription fee coefficient
        delta_h : float
            Hospital discount factor
        
        Returns
        -------
        float
            Expected profit for this contract type
        """
        
        if contract == 'c_Y':
            # Usage-based contract: P(c_Y) = P_0 + δ_h · P_m(v)
            payment_L = p_0 + delta_h * p_m_coeff * v_low_bias
            
            # Only low-bias hospitals accept c_Y
            # High-bias reject due to upfront fee
            expected_revenue = self.p_L * payment_L
            
        elif contract == 'c_S':
            # Subscription contract: P(c_S) = δ_h · γ · v
            payment_L = delta_h * gamma * v_low_bias
            payment_H = delta_h * gamma * v_high_bias
            
            # Both types accept c_S
            expected_revenue = (
                self.p_L * payment_L +
                (1 - self.p_L) * payment_H
            )
        
        else:
            raise ValueError(f"Unknown contract: {contract}")
        
        # Expected profit = revenue - cost
        expected_profit = expected_revenue - self.params.c_cost
        
        return expected_profit
    
    # =========================================================================
    # PURE STRATEGY CONTRACT SELECTION
    # =========================================================================
    
    def decide_contract_pure_strategy(
        self,
        v_low_bias: float,
        v_high_bias: float,
        p_0: float,
        p_m_coeff: float,
        gamma: float,
        delta_h: float,
        return_profits: bool = False
    ) -> Tuple[str, float, float] or str:
        """
        Make pure strategy contract choice using threshold rule.
        
        Decision Rule (Deterministic Threshold):
            1. Calculate E[π(c_Y | p_L)]
            2. Calculate E[π(c_S | p_L)]
            3. If E[π(c_Y)] >= E[π(c_S)]:  Choose c_Y
            4. Else:                        Choose c_S
        
        This pure strategy is optimal because:
            - Hospitals don't learn firm's choice (fixed p_Y belief)
            - Firm has no benefit from randomization
            - Aligns with analytical solution
        
        Parameters
        ----------
        v_low_bias : float
            Low-bias perceived value
        v_high_bias : float
            High-bias perceived value
        p_0 : float
            Upfront fee
        p_m_coeff : float
            Usage-based coefficient
        gamma : float
            Subscription coefficient
        delta_h : float
            Hospital discount factor
        return_profits : bool, optional
            If True, return (contract, profit_cY, profit_cS)
        
        Returns
        -------
        str or Tuple[str, float, float]
            Contract choice or (contract, profit_cY, profit_cS)
        """
        
        profit_cY = self.expected_profit_from_contract(
            'c_Y', v_low_bias, v_high_bias, p_0, p_m_coeff, gamma, delta_h
        )
        
        profit_cS = self.expected_profit_from_contract(
            'c_S', v_low_bias, v_high_bias, p_0, p_m_coeff, gamma, delta_h
        )
        
        # Pure strategy: choose maximum
        contract = 'c_Y' if profit_cY >= profit_cS else 'c_S'
        
        # Record choice
        self.contract_history.append(contract)
        
        if return_profits:
            return contract, profit_cY, profit_cS
        else:
            return contract
    
    def get_threshold_p_L(
        self,
        v_low_bias: float,
        v_high_bias: float,
        p_0: float,
        p_m_coeff: float,
        gamma: float,
        delta_h: float
    ) -> float:
        """
        Calculate the indifference threshold p_L* where firm is indifferent
        between c_Y and c_S.
        
        At p_L*, E[π(c_Y | p_L*)] = E[π(c_S | p_L*)]
        
        This is useful for understanding the equilibrium structure.
        
        Theory:
            p_L · Payment_L^cY - C = p_L · Payment_L^cS + (1-p_L) · Payment_H^cS - C
            p_L · (Payment_L^cY - Payment_L^cS) = (1-p_L) · Payment_H^cS
            p_L · (Payment_L^cY - Payment_L^cS) = Payment_H^cS - p_L · Payment_H^cS
            p_L · (Payment_L^cY - Payment_L^cS + Payment_H^cS) = Payment_H^cS
        
        Parameters
        ----------
        v_low_bias : float
            Low-bias perceived value
        v_high_bias : float
            High-bias perceived value
        p_0 : float
            Upfront fee
        p_m_coeff : float
            Usage-based coefficient
        gamma : float
            Subscription coefficient
        delta_h : float
            Hospital discount factor
        
        Returns
        -------
        float
            Threshold p_L* ∈ (0, 1)
        """
        
        # Calculate revenues
        payment_L_cY = p_0 + delta_h * p_m_coeff * v_low_bias
        payment_L_cS = delta_h * gamma * v_low_bias
        payment_H_cS = delta_h * gamma * v_high_bias
        
        # Solve for p_L*:
        # p_L * (payment_L_cY - payment_L_cS) = (1-p_L) * payment_H_cS
        # p_L * (payment_L_cY - payment_L_cS + payment_H_cS) = payment_H_cS
        
        numerator = payment_H_cS
        denominator = payment_L_cY - payment_L_cS + payment_H_cS
        
        if abs(denominator) < 1e-10:
            return 0.5  # Default if no clear indifference
        
        p_L_star = numerator / denominator
        return np.clip(p_L_star, 0, 1)
    
    # =========================================================================
    # PAYOFF REALIZATION AND HISTORY
    # =========================================================================
    
    def record_adoption(self, contract: str, revenue: float):
        """
        Record a hospital adoption and update payoff.
        
        Parameters
        ----------
        contract : str
            Contract type ('c_Y' or 'c_S')
        revenue : float
            Revenue from this adoption
        """
        net_profit = revenue - self.params.c_cost
        self.payoff += net_profit
        
        if contract == 'c_Y':
            self.num_cY_adopted += 1
            self.revenue_from_cY += revenue
        else:
            self.num_cS_adopted += 1
            self.revenue_from_cS += revenue
    
    def reset(self):
        """Reset firm to initial state."""
        initial_p_L = self.p_L_history[0]
        self.p_L = initial_p_L
        self.contract_history = []
        self.payoff = 0.0
        self.revenue_from_cY = 0.0
        self.revenue_from_cS = 0.0
        self.num_cY_adopted = 0
        self.num_cS_adopted = 0
        self.p_L_history = [initial_p_L]
    
    def __repr__(self) -> str:
        return (
            f"Firm(id={self.agent_id}, "
            f"p_L={self.p_L:.2f}, "
            f"payoff={self.payoff:.2f})"
        )
