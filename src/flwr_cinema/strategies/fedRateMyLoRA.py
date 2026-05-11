r"""Rate-My-LoRA aggregation strategy for federated learning.

Reference
---------
He, X., Shi, H., Han, L., Tan, C., Liu, B., Xu, Z., Ye, M.,
Axel, L., Li, K., & Metaxas, D. (2025).
*Rate-My-LoRA: Efficient and Adaptive Federated Model Tuning
for Cardiac MRI Segmentation.* ISBI 2025.

Algorithm 1 — The proposed Rate-My-LoRA method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Per communication round t::

    # --- Training phase (aggregate_train) ---
    # Algo ln 7-9: clients train locally, upload adapters B^c_t A^c_t
    for each c in C (in parallel):
        B^c_t A^c_t ← TrainLocal(M + Σ_{k∈C} B^k_{t-1} A^k_{t-1})

    # Algo ln 8 (implicit): server collects adapters from all clients

    # --- Preliminary aggregation ---
    # Paper: "the adapters are first aggregated with EQUAL WEIGHTS
    #         and distributed to each client"
    # ⇒ _equal_weight_aggregate()

    # --- Evaluation phase (configure_evaluate → aggregate_evaluate) ---
    # Algo ln 11-12: distribute preliminary model, evaluate per client
    for each c in C (in parallel):
        P^c_t ← Evaluate(M^A_t) on client c's validation set

    # --- Adaptive re-weighting ---
    # Eqn 1: w_t(c) = 1-λ if ∃i∈C s.t. P^i_t < P^i_{t-1} AND P^c_t > P^c_{t-1}
    #                   1  otherwise
    # ⇒ aggregate_evaluate(): compute has_drop, compute w_t(c) per client

    # Eqn 2: W_t = W_{t-1} + (1 / Σ|D^c|) * Σ w_t(c) * |D^c| * B^c_t A^c_t
    # ⇒ _adaptive_aggregate()

    # Algo ln 15: distribute B^c_{t+1}, A^c_{t+1} for next round
    # ⇒ configure_train() injects _reaggregated_arrays into arrays arg

    # Algo ln 16: λ = λ × 0.95  (diminishing schedule, unconditional)
    # ⇒ aggregate_evaluate(): self.lambda_ *= self.lambda_decay

    # Paper: "When λ=0, method reverts to FedAVG"
"""
from __future__ import annotations

from collections.abc import Iterable
import logging

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.common.logger import log

from flwr_cinema.strategies.common import CinemaStrategyMixin


# ==============================================================================
# Paper hyper-parameter defaults (Section "The proposed Rate-My-LoRA method")
# ==============================================================================
#   λ (lambda)  = 0.2           → penalty factor for overfitting clients
#   λ decay     = 0.95 / round  → "reducing it by 5% each communication round"
#   LoRA rank   = 16, 32, 32    → proportional to |D^c| (set per-client, not here)


class RateMyLoRA(FedAvg):
    """FedAvg variant implementing the Rate-My-LoRA adaptive re-weighting.

    Paper sections
    --------------
    - ``Weight computation``:  Eqn 1  (w_t(c))
    - ``Re-aggregation``:      Eqn 2  (W_t)
    - ``Lambda schedule``:     λ ← λ × 0.95 each round
    - ``Preliminary phase``:   adapters aggregated with EQUAL weights before eval

    Parameters
    ----------
    lambda_init : float
        Initial penalisation factor λ (paper default: 0.2).
    lambda_decay : float
        Multiplicative decay per round (paper default: 0.95).
    primary_metric_name : str
        Metric used for per-client accuracy tracking (paper: Dice score).
    primary_metric_direction : {"max", "min"}
        ``"max"`` if higher is better (Dice), ``"min"`` if lower is better (loss).
    """

    def __init__(
        self,
        lambda_init: float = 0.2,
        lambda_decay: float = 0.95,
        primary_metric_name: str = "mean_dice_score",
        primary_metric_direction: str = "max",
        **kwargs,
    ):
        kwargs.setdefault("weighted_by_key", "num_samples")
        super().__init__(**kwargs)

        # --- Penalty factor λ (Eqn 1) ---
        self.lambda_: float = float(lambda_init)

        # --- Diminishing schedule: λ ← λ × 0.95 each round ---
        self.lambda_decay: float = float(lambda_decay)

        # --- Metric tracking for P^c_t and P^c_{t-1} ---
        self.primary_metric_name: str = primary_metric_name
        self.metric_direction: str = primary_metric_direction

        # --- Paper: P^c_{t-1} — previous round validation accuracy per client ---
        self._previous_accuracy: dict[str, float] = {}

        # --- Paper: B^c_t A^c_t — raw per-client LoRA adapters (Eqn 2) ---
        self._stored_adapters: list[tuple[str, ArrayRecord, int]] = []

        # --- Paper: W_t — the re-aggregated global model (Eqn 2) ---
        self._reaggregated_arrays: ArrayRecord | None = None

    # ------------------------------------------------------------------
    # Flower lifecycle hooks — mapped to Algorithm 1
    # ------------------------------------------------------------------

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        # Algo ln 15: "Distribute B^c_{t+1}, A^c_{t+1}"
        # From round 2 onwards, replace the default global model with the
        # adaptively re-aggregated model W_t computed in the previous round.
        if server_round > 1 and self._reaggregated_arrays is not None:
            arrays = self._reaggregated_arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # Algo ln 6-9: "Collect adapter from client c"
        # Store raw B^c_t A^c_t adapters for later re-aggregation (Eqn 2).
        replies_list = list(replies)
        self._stored_adapters = []
        for msg in replies_list:
            cid = self._resolve_client_id(msg)
            metrics = msg.content.metric_records.get("metrics", MetricRecord())

            # |D^c| — dataset size used as weight in Eqn 2
            num_samples = int(metrics.get("num_samples", 0))

            # B^c_t A^c_t — per-client LoRA adapter
            arrays = msg.content.array_records.get("model_params")
            if arrays is not None:
                self._stored_adapters.append((cid, arrays, num_samples))

        # --- Preliminary aggregation ---
        # Paper: "the adapters are first aggregated with EQUAL WEIGHTS
        #         and distributed to each client, which then evaluates
        #         the model on its local validation set"
        # This is the model clients evaluate to produce P^c_t.
        if self._stored_adapters:
            preliminary = self._equal_weight_aggregate()
        else:
            preliminary = None

        # Delegate metric bookkeeping to FedAvg (not used for adaptive weights)
        _, agg_metrics = super().aggregate_train(server_round, replies_list)
        return preliminary, agg_metrics

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        # Algo ln 11: "Evaluate M^A_t on client c" (in parallel)
        # The `arrays` parameter here is the preliminary equal-weight model
        # returned by aggregate_train() above.
        return FedAvg.configure_evaluate(self, server_round, arrays, config, grid)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        # Algo ln 11-12: collect P^c_t from each client
        #      "P^c_t ← Evaluate M^A_t on client c and return the evaluation metric"
        replies_list = list(replies)

        # P^c_t — current round validation accuracy per client
        current_accuracy: dict[str, float] = {}
        for msg in replies_list:
            cid = self._resolve_client_id(msg)
            metrics = msg.content.metric_records.get("metrics", MetricRecord())
            value = metrics.get(self.primary_metric_name)
            if value is not None:
                current_accuracy[cid] = float(value)

        if current_accuracy:
            # ------------------------------------------------------------------
            # Algo ln 13 (implicit): compare P^c_t vs P^c_{t-1}
            # Only clients present in BOTH rounds participate in drop detection.
            # ------------------------------------------------------------------
            common_clients = {
                cid for cid in current_accuracy
                if cid in self._previous_accuracy
            }
            direction = getattr(self, "metric_direction", "max")

            # --- Eqn 1: "∃ i ∈ C s.t. P^i_t < P^i_{t-1}" ---
            # has_drop = True iff ANY client's accuracy decreased this round.
            # This is the "guard" of the conditional in Eqn 1.
            # In text: "When such performance drop is reported, the server
            #           adaptively penalizes the adapter from other clients."
            if direction == "max":
                has_drop = any(
                    current_accuracy[cid] < self._previous_accuracy[cid]
                    for cid in common_clients
                )
            else:  # "min" — lower is better (e.g. loss, MAE)
                has_drop = any(
                    current_accuracy[cid] > self._previous_accuracy[cid]
                    for cid in common_clients
                )

            # ------------------------------------------------------------------
            # Eqn 1: compute w_t(c) for each client c
            #
            #         ⎧ 1-λ   if ∃i∈C s.t. P^i_t < P^i_{t-1}
            # w_t(c) = ⎨                AND  P^c_t > P^c_{t-1}
            #         ⎩ 1     otherwise
            #
            # Interpretation (from paper): "adaptively penalize clients whose
            # adapters are contributing to the degradation of generalizability"
            # i.e., clients that OVERFIT to local data while others suffer.
            # ------------------------------------------------------------------
            weights: dict[str, float] = {}
            for cid in current_accuracy:
                if cid not in self._previous_accuracy:
                    weights[cid] = 1.0  # new client, no P^c_{t-1} history
                    continue

                # "P^c_t > P^c_{t-1}" — client c improved
                if direction == "max":
                    improved = current_accuracy[cid] > self._previous_accuracy[cid]
                else:
                    improved = current_accuracy[cid] < self._previous_accuracy[cid]

                # Eqn 1: w_t(c) = 1-λ iff (has_drop AND improved)
                weights[cid] = (1.0 - self.lambda_) if (has_drop and improved) else 1.0

                # In-text: "When such performance drop is reported, the server
                #           adaptively penalizes the adapter from other clients,
                #           as they fail to generalize."
                log(logging.INFO, f"Client {cid}: current weight {weights[cid]:.3f}")

            # ------------------------------------------------------------------
            # Eqn 2 — Re-aggregation with adaptive weights
            #
            #   W_t = W_{t-1} + (1 / Σ|D^c|) * Σ w_t(c) * |D^c| * B^c_t A^c_t
            #         └───────────┬───────────┘   └────────────┬────────────┘
            #              raw dataset size sum        per-client contribution
            #              (denominator, NOT Σ w·|D^c|)
            #
            # This replaces the model for the NEXT round's configure_train().
            # ------------------------------------------------------------------
            if self._stored_adapters:
                self._reaggregated_arrays = self._adaptive_aggregate(weights)
            else:
                # No adapters stored → clear stale reference so the next round
                # doesn't distribute an outdated W_{t-1}.
                self._reaggregated_arrays = None

            # Update P^c_{t-1} ← P^c_t for the next round's drop detection
            self._previous_accuracy = current_accuracy

        # ----------------------------------------------------------------------
        # Algo ln 16: λ = λ × 0.95
        # Paper: "To ensure convergence, we apply a diminishing schedule to λ,
        #         reducing it by 5% each communication round."
        # UNCONDITIONAL — must decay even when no metrics are reported.
        # When λ → 0, w_t(c) → 1 ∀c, and the method reverts to FedAVG.
        # ----------------------------------------------------------------------
        self.lambda_ *= self.lambda_decay

        # Clear adapters unconditionally to prevent stale reuse across rounds
        self._stored_adapters = []

        # If no accuracy data was collected this round, invalidate any stale
        # W_{t-1} so configure_train() falls back to the freshest preliminary
        # model rather than distributing an outdated re-aggregation.
        if not current_accuracy:
            self._reaggregated_arrays = None

        return super().aggregate_evaluate(server_round, replies_list)

    def get_final_arrays(self) -> ArrayRecord | None:
        """Return W_t (Eqn 2), or None if re-aggregation hasn't run yet.

        Public accessor for server_app so it can retrieve the final
        adaptively-re-aggregated model without reaching into ``_reaggregated_arrays``.
        """
        return self._reaggregated_arrays

    # ==================================================================
    # Helpers — direct implementations of paper formulas
    # ==================================================================

    @staticmethod
    def _resolve_client_id(msg: Message) -> str:
        """Extract logical client ID from Flower message metadata."""
        config = msg.content.config_records.get("config", ConfigRecord())
        cid = config.get("client_id")
        return str(cid) if cid is not None else str(msg.metadata.src_node_id)

    def _equal_weight_aggregate(self) -> ArrayRecord:
        """Preliminary aggregation phase.

        Paper: *"the adapters are first aggregated with EQUAL WEIGHTS
        and distributed to each client, which then evaluates the model
        on its local validation set and reports accuracy to the server."*

        Formula: (1/|C|) Σ_{c∈C} B^c_t A^c_t

        Each client's adapter contributes equally — dataset size |D^c|
        is NOT used here.  This is the model clients evaluate to
        produce P^c_t for the adaptive re-weighting phase.

        Notes
        -----
        Uses ``keep_input=True`` on ``to_torch_state_dict()`` so the
        same ArrayRecord objects remain intact for the subsequent
        :meth:`_adaptive_aggregate` call.
        """
        import torch

        aggregated: dict[str, torch.Tensor] = {}

        # |C| — number of participating clients
        n = max(len(self._stored_adapters), 1)

        for _cid, arrays, _num_samples in self._stored_adapters:
            # B^c_t A^c_t — per-client LoRA adapter
            # keep_input=True: preserve ArrayRecord for _adaptive_aggregate
            state = arrays.to_torch_state_dict(keep_input=True)

            # Accumulate: (1/|C|) · Σ B^c_t A^c_t
            for key, tensor in state.items():
                if key not in aggregated:
                    aggregated[key] = tensor.float() / n
                else:
                    aggregated[key] += tensor.float() / n

        return ArrayRecord.from_torch_state_dict(aggregated, keep_input=False)

    def _adaptive_aggregate(self, weights: dict[str, float]) -> ArrayRecord:
        """Final re-aggregation with adaptive weights w_t(c).

        Paper Eqn 2::

            W_t = W_{t-1} + (1 / Σ|D^c|) * Σ w_t(c) * |D^c| * B^c_t A^c_t
                              └─────┬─────┘   └──────────┬──────────────┘
                          denominator =         per-client contribution
                          sum of raw            (adaptive weight ×
                          dataset sizes)        dataset size × adapter)

        **Denominator is Σ|D^c| (raw dataset sizes), NOT Σ w_t(c)·|D^c|.**

        This distinction matters: when some clients are penalised
        (w_t(c) = 1-λ), the denominator MUST still be the total dataset
        size, preserving the correct update magnitude as specified in
        the paper.  Using Σ w·|D^c| as denominator would artificially
        inflate penalised clients' contributions.
        """
        import torch

        # Numerator accumulator: Σ w_t(c) * |D^c| * B^c_t A^c_t
        aggregated: dict[str, torch.Tensor] = {}

        # Denominator: Σ |D^c|  (raw dataset sizes, NOT weighted)
        raw_total: float = 0.0

        for cid, arrays, num_samples in self._stored_adapters:
            # B^c_t A^c_t — per-client LoRA adapter
            state = arrays.to_torch_state_dict(keep_input=False)

            # |D^c| — client dataset size (≥ 1)
            n = max(num_samples, 1)

            # w_t(c) * |D^c| — per-client contribution weight (Eqn 2 numerator term)
            w = weights.get(cid, 1.0) * n

            # Accumulate denominator: Σ |D^c|
            raw_total += n

            # Accumulate numerator: Σ w_t(c) * |D^c| * B^c_t A^c_t
            for key, tensor in state.items():
                if key not in aggregated:
                    aggregated[key] = tensor.float() * w
                else:
                    aggregated[key] += tensor.float() * w

        # Normalise: (numerator) / Σ|D^c|  (Eqn 2)
        if raw_total > 0:
            for key in aggregated:
                aggregated[key] /= raw_total

        return ArrayRecord.from_torch_state_dict(aggregated, keep_input=False)


# ==============================================================================
# Strategy factory class — aggregates RateMyLoRA with CinemaStrategyMixin
# (checkpointing, early stopping)
# ==============================================================================


class RateMyLoraStrategy(CinemaStrategyMixin, RateMyLoRA):
    """Rate-My-LoRA + CinemaStrategyMixin (checkpointing, early stopping).

    Multiple-inheritance MRO: CinemaStrategyMixin → RateMyLoRA → FedAvg.

    This class overrides ``configure_evaluate`` and ``aggregate_evaluate``
    to force evaluation EVERY round (as required by Algorithm 1 of the
    paper), while still respecting the ``eval_steps`` interval for
    checkpointing and early-stopping decisions.
    """

    def __init__(self, **kwargs):
        # Pop RateMyLoRA params before passing to MRO super chain
        lambda_init = kwargs.pop("lambda_init", 0.2)
        lambda_decay = kwargs.pop("lambda_decay", 0.95)
        super().__init__(lambda_init=lambda_init, lambda_decay=lambda_decay, **kwargs)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        # Algo ln 11: evaluate MUST run every round
        # (Rate-My-LoRA needs P^c_t per round for drop detection).
        # Override CinemaStrategyMixin's step-gating here — it would
        # skip evaluation rounds, breaking the adaptive weighting.

        if getattr(self, "early_stopping_triggered", False):
            return []

        # Store preliminary model reference for checkpointing
        self._latest_arrays = arrays
        config["total_optimizer_steps"] = self.total_optimizer_steps

        return FedAvg.configure_evaluate(self, server_round, arrays, config, grid)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Two-phase evaluation aggregation.

        Phase A (always runs — RateMyLoRA requirement):
            Algo ln 11-16: collect P^c_t, compute w_t(c) via Eqn 1,
            re-aggregate via Eqn 2, decay λ.

        Phase B (gated by eval_steps — CinemaStrategyMixin requirement):
            Check early stopping, save checkpoints.
        """

        # --- Phase A: Algo ln 11-16 (always runs) ---
        # RateMyLoRA.aggregate_evaluate handles:
        #   - collecting P^c_t
        #   - computing w_t(c) via Eqn 1
        #   - re-aggregating via Eqn 2 → _reaggregated_arrays
        #   - decaying λ: λ ← λ × 0.95
        result = RateMyLoRA.aggregate_evaluate(self, server_round, replies)

        # After re-aggregation, update _latest_arrays so any checkpoint
        # saved by CinemaStrategyMixin captures W_t (Eqn 2), NOT the
        # preliminary equal-weight model.
        if self._reaggregated_arrays is not None:
            self._latest_arrays = self._reaggregated_arrays

        # Track number of completed evaluation rounds.
        # Rate-My-LoRA evaluates every round (not just gated ones),
        # so increment here BEFORE the step-gating check.
        if result is not None and self.primary_metric_name in result:
            self._eval_count += 1

        # --- Phase B: Early stopping (gated by eval_steps) ---
        # CinemaStrategyMixin._apply_early_stopping runs checkpointing
        # and early-stopping logic, but only every eval_steps rounds.
        # eval_count_increment=False because we already incremented _eval_count above.
        if self.eval_steps > 0 and self.total_optimizer_steps > 0:
            steps_since = self.total_optimizer_steps - self.last_eval_step
            if steps_since >= self.eval_steps:
                self.last_eval_step = (
                    self.total_optimizer_steps // self.eval_steps
                ) * self.eval_steps
                log(
                    logging.INFO,
                    f"Eval steps reached ({steps_since} since last eval): "
                    f"applying early stopping check.",
                )
                self._apply_early_stopping(
                    server_round, result, eval_count_increment=False
                )

        return result
