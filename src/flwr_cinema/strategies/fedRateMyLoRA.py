r"""Rate-My-LoRA aggregation strategy for federated learning.

Implements Algorithm 1 from He et al. (ISBI 2025).

Clients train local LoRA adapters, the server detects overfitting via
per-client accuracy drops, and adaptively re-weights contributions.

    w_t(c) = 1-lambda if any client dropped AND client c improved, else 1    (Eqn 1)
    W_t = W_{t-1} + (1 / sum|Dc|) * sum w_t(c) * |Dc| * Bc_t Ac_t           (Eqn 2)

lambda decays by 0.95 each round. When lambda reaches 0, this reduces to FedAvg.
"""
from __future__ import annotations

from collections.abc import Iterable
import logging

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.common.logger import log

from flwr_cinema.strategies.common import CinemaStrategyMixin


# Paper hyper-parameter defaults: lambda=0.2, decay=0.95/round, rank per-client.


class RateMyLoRA(FedAvg):
    """FedAvg variant implementing Rate-My-LoRA adaptive re-weighting.

    Per-round: equal-weight preliminary aggregation, per-client evaluation
    to detect accuracy drops (Eqn 1), then adaptive re-aggregation (Eqn 2).
    Lambda decays by lambda_decay each round.

    Args:
        lambda_init: Initial penalty factor (paper default: 0.2).
        lambda_decay: Multiplicative decay per round (paper default: 0.95).
        primary_metric_name: Metric for per-client accuracy tracking (paper: Dice).
        primary_metric_direction: "max" if higher is better, "min" otherwise.
        warmup_rounds: Rounds to skip adaptive re-weighting. During warmup,
            behaves like plain FedAvg while collecting accuracy baselines.
            Default 0 (adaptive from round 1).
    """

    def __init__(
        self,
        lambda_init: float = 0.2,
        lambda_decay: float = 0.95,
        primary_metric_name: str = "mean_dice_score",
        primary_metric_direction: str = "max",
        warmup_rounds: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("weighted_by_key", "num_samples")
        super().__init__(**kwargs)

        # Penalty factor lambda (Eqn 1).
        self.lambda_: float = float(lambda_init)
        # Diminishing schedule: lambda *= lambda_decay each round.
        self.lambda_decay: float = float(lambda_decay)
        # Warmup: skip adaptive re-weighting during first N rounds.
        self.warmup_rounds: int = warmup_rounds
        # Metric tracking for P^c_t and P^c_{t-1}.
        self.primary_metric_name: str = primary_metric_name
        self.metric_direction: str = primary_metric_direction
        # P^c_{t-1}: previous round validation accuracy per client.
        self._previous_accuracy: dict[str, float] = {}
        # B^c_t A^c_t: raw per-client LoRA adapters (Eqn 2).
        self._stored_adapters: list[tuple[str, ArrayRecord, int]] = []
        # W_t: re-aggregated global model (Eqn 2).
        self._reaggregated_arrays: ArrayRecord | None = None

    # -- Flower lifecycle hooks -------------------------------------------------

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        # After warmup: distribute W_t (Eqn 2) instead of default FedAvg model.
        if server_round > self.warmup_rounds and self._reaggregated_arrays is not None:
            arrays = self._reaggregated_arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # Collect B^c_t A^c_t and |D^c| from each client for Eqn 2.
        replies_list = list(replies)
        self._stored_adapters = []
        for msg in replies_list:
            cid = self._resolve_client_id(msg)
            metrics = msg.content.metric_records.get("metrics", MetricRecord())
            num_samples = int(metrics.get("num_samples", 0))
            arrays = msg.content.array_records.get("model_params")
            if arrays is not None:
                self._stored_adapters.append((cid, arrays, num_samples))

        # Preliminary equal-weight aggregation (model clients evaluate for P^c_t).
        preliminary = self._equal_weight_aggregate() if self._stored_adapters else None
        _, agg_metrics = super().aggregate_train(server_round, replies_list)
        return preliminary, agg_metrics

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        # Evaluate M^A_t on each client (preliminary equal-weight model from aggregate_train).
        return FedAvg.configure_evaluate(self, server_round, arrays, config, grid)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        # Collect P^c_t — current round validation accuracy per client.
        replies_list = list(replies)
        current_accuracy: dict[str, float] = {}
        for msg in replies_list:
            cid = self._resolve_client_id(msg)
            metrics = msg.content.metric_records.get("metrics", MetricRecord())
            value = metrics.get(self.primary_metric_name)
            if value is not None:
                current_accuracy[cid] = float(value)

        if current_accuracy:
            # Skip adaptive re-weighting during warmup — collect P^c_t only.
            if server_round > self.warmup_rounds:
                direction = getattr(self, "metric_direction", "max")

                # has_drop: any client's accuracy decreased this round.
                common_clients = {
                    cid for cid in current_accuracy
                    if cid in self._previous_accuracy
                }
                if direction == "max":
                    has_drop = any(
                        current_accuracy[cid] < self._previous_accuracy[cid]
                        for cid in common_clients
                    )
                else:
                    has_drop = any(
                        current_accuracy[cid] > self._previous_accuracy[cid]
                        for cid in common_clients
                    )

                # Eqn 1: w_t(c) = 1-lambda if has_drop AND client improved, else 1.
                weights: dict[str, float] = {}
                for cid in current_accuracy:
                    if cid not in self._previous_accuracy:
                        weights[cid] = 1.0
                        continue
                    if direction == "max":
                        improved = current_accuracy[cid] > self._previous_accuracy[cid]
                    else:
                        improved = current_accuracy[cid] < self._previous_accuracy[cid]
                    weights[cid] = (1.0 - self.lambda_) if (has_drop and improved) else 1.0
                    log(logging.INFO, f"Client {cid}: current weight {weights[cid]:.3f}")

                # Eqn 2: W_t = W_{t-1} + (1 / sum|Dc|) * sum w_t(c) * |Dc| * Bc_t Ac_t
                if self._stored_adapters:
                    self._reaggregated_arrays = self._adaptive_aggregate(weights)
                else:
                    self._reaggregated_arrays = None

            # P^c_{t-1} ← P^c_t (always, including warmup).
            self._previous_accuracy = current_accuracy

        # lambda *= 0.95 each round (diminishing schedule).
        if server_round > self.warmup_rounds:
            self.lambda_ *= self.lambda_decay

        self._stored_adapters = []
        if not current_accuracy:
            self._reaggregated_arrays = None
        return super().aggregate_evaluate(server_round, replies_list)

    def get_final_arrays(self) -> ArrayRecord | None:
        """Return W_t (Eqn 2), or None if re-aggregation hasn't run yet."""
        return self._reaggregated_arrays

    # -- Helpers (paper formulas) ----------------------------------------------

    @staticmethod
    def _resolve_client_id(msg: Message) -> str:
        """Extract logical client ID from Flower message metadata."""
        config = msg.content.config_records.get("config", ConfigRecord())
        cid = config.get("client_id")
        return str(cid) if cid is not None else str(msg.metadata.src_node_id)

    def _equal_weight_aggregate(self) -> ArrayRecord:
        """Preliminary equal-weight aggregation: (1/|C|) * sum Bc_t Ac_t.

        Uses keep_input=True so ArrayRecords survive for _adaptive_aggregate.
        """
        import torch

        aggregated: dict[str, torch.Tensor] = {}
        n = max(len(self._stored_adapters), 1)

        for _cid, arrays, _num_samples in self._stored_adapters:
            # keep_input=True: preserve ArrayRecord for _adaptive_aggregate
            state = arrays.to_torch_state_dict(keep_input=True)
            for key, tensor in state.items():
                if key not in aggregated:
                    aggregated[key] = tensor.float() / n
                else:
                    aggregated[key] += tensor.float() / n

        return ArrayRecord.from_torch_state_dict(aggregated, keep_input=False)

    def _adaptive_aggregate(self, weights: dict[str, float]) -> ArrayRecord:
        """Re-aggregation with adaptive weights.

        Eqn 2: W_t = W_{t-1} + (1 / sum|Dc|) * sum w_t(c) * |Dc| * Bc_t Ac_t

        Denominator is sum|Dc| (raw dataset sizes), not sum w_t(c) * |Dc|.
        """
        import torch

        # Numerator: sum w_t(c) * |Dc| * Bc_t Ac_t
        aggregated: dict[str, torch.Tensor] = {}
        # Denominator: sum |Dc| (raw dataset sizes)
        raw_total: float = 0.0

        for cid, arrays, num_samples in self._stored_adapters:
            state = arrays.to_torch_state_dict(keep_input=False)
            n = max(num_samples, 1)
            w = weights.get(cid, 1.0) * n
            raw_total += n
            for key, tensor in state.items():
                if key not in aggregated:
                    aggregated[key] = tensor.float() * w
                else:
                    aggregated[key] += tensor.float() * w

        if raw_total > 0:
            for key in aggregated:
                aggregated[key] /= raw_total

        return ArrayRecord.from_torch_state_dict(aggregated, keep_input=False)


# Strategy factory: RateMyLoRA + CinemaStrategyMixin (checkpointing, early stopping).


class RateMyLoraStrategy(CinemaStrategyMixin, RateMyLoRA):
    """Rate-My-LoRA + CinemaStrategyMixin (checkpointing, early stopping).

    MRO: CinemaStrategyMixin -> RateMyLoRA -> FedAvg.
    Forces evaluation every round (required by Algorithm 1) while gating
    checkpointing and early stopping on eval_steps.
    """

    def __init__(self, **kwargs):
        # Pop RateMyLoRA params before passing to MRO super chain
        lambda_init = kwargs.pop("lambda_init", 0.2)
        lambda_decay = kwargs.pop("lambda_decay", 0.95)
        warmup_rounds = kwargs.pop("warmup_rounds", 0)
        super().__init__(
            lambda_init=lambda_init,
            lambda_decay=lambda_decay,
            warmup_rounds=warmup_rounds,
            **kwargs,
        )

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        # Evaluate every round — drop detection needs P^c_t per round.
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
        """Phase A: RateMyLoRA adaptive re-weighting (always). Phase B: early stopping (gated)."""

        # Phase A: Eqn 1-2 + lambda decay.
        result = RateMyLoRA.aggregate_evaluate(self, server_round, replies)

        # Update _latest_arrays so checkpoints capture W_t (Eqn 2), not preliminary model.
        if self._reaggregated_arrays is not None:
            self._latest_arrays = self._reaggregated_arrays

        # Track number of completed evaluation rounds.
        # Rate-My-LoRA evaluates every round (not just gated ones),
        # so increment here BEFORE the step-gating check.
        if result is not None and self.primary_metric_name in result:
            self._eval_count += 1

        # Phase B: early stopping gated by eval_steps (already incremented _eval_count above).
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
