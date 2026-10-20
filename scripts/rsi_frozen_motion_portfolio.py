"""Single measured boundary selection; frozen expert memory, no latent blending.

Every expert warms on the actual prefix. One neural owner is fixed from frame20
through299. This adapter contains no world/robot/transport or promotion handle.
"""

import hashlib


class FrozenMotionPortfolio:
    def __init__(self, router, experts, *, minimum_route_confidence=0.75):
        if (
            len(experts) != router.expert_count
            or not 2 <= len(experts) <= 16
            or minimum_route_confidence != 0.75
        ):
            raise ValueError("fixed bounded roster and preregistered confidence required")
        for model in [router, *experts]:
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError("explicitly frozen eval-only models required")
        self.router, self.experts = router, list(experts)
        self.minimum_route_confidence = minimum_route_confidence
        self.frame = 0
        self.memories = [None] * len(experts)
        self.owner = None
        self.audits = []
        self.prefix_hidden = []

    def network_call_counts(self):
        if self.frame != 300:
            raise ValueError("complete owned portfolio stream required for counters")
        return dict(
            policy_invocations=300,
            intention_network_calls=20 * len(self.experts) + 280,
            router_network_calls=1,
        )

    def sample(self, x, hidden, *, generator, stochastic):
        import torch

        from rosclaw_soccer.rsi.motion_capability_router import motion_routing_features

        if type(stochastic) is not bool or stochastic:
            raise ValueError("frozen deterministic portfolio evaluation only")
        if x.shape != (1, 1, 108) or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("one current owned observation required")
        if hidden is None:
            if self.frame not in (0, 300) or float(x[0, 0, 0]) != 1.0:
                raise ValueError("no mid-body memory reset")
            self.frame, self.owner, self.prefix_hidden = 0, None, []
            self.memories = [None] * len(self.experts)
            self.audits.append(
                dict(
                    stream_index=len(self.audits),
                    router_calls=0,
                    minimum_route_confidence=self.minimum_route_confidence,
                )
            )
        elif self.frame >= 300:
            raise ValueError("full H300 stream already complete")
        else:
            expected = self.memories[0 if self.frame <= 20 else self.owner]
            if expected is None or not torch.equal(hidden, expected):
                raise ValueError("caller memory differs from actual preceding owner")
        if self.frame < 20:
            outputs = []
            for i, model in enumerate(self.experts):
                row = model.sample(x, self.memories[i], generator=generator, stochastic=False)
                self.memories[i] = row["hidden"]
                outputs.append(row)
            self.prefix_hidden.append(
                torch.stack([h[0, 0].detach().cpu() for h in self.memories]).numpy().copy()
            )
            result = outputs[0]
        else:
            if self.frame == 20:
                import numpy as np

                features = motion_routing_features(x[0, 0].detach().cpu().numpy())
                logits, success = self.router(torch.from_numpy(features)[None])
                probabilities = logits.softmax(-1)[0]
                proposed = int(probabilities.argmax())
                confidence = float(probabilities[proposed])
                self.owner = (
                    proposed if proposed == 0 or confidence >= self.minimum_route_confidence else 0
                )
                self.audits[-1].update(
                    router_calls=1,
                    gate_features=features.tolist(),
                    proposed_owner=proposed,
                    selected_owner=self.owner,
                    routing_confidence=confidence,
                    uncalibrated_success_scores=success.sigmoid()[0].detach().cpu().tolist(),
                    prefix_all_expert_memory_sha256=hashlib.sha256(
                        np.stack(self.prefix_hidden).tobytes()
                    ).hexdigest(),
                    no_new_hardware_or_promotion_authority=True,
                )
            result = self.experts[self.owner].sample(
                x, self.memories[self.owner], generator=generator, stochastic=False
            )
            self.memories[self.owner] = result["hidden"]
        self.frame += 1
        return result
