"""Qualify shadow query/actual-action history against original Foundation prefix.

This is a prerequisite, not evidence that teacher intervention recovers a student.
All 20 queries consume their original advice; a separate codec recomputes all930
history entries from measured q/v and the actually committed previous29 action.
"""


class ShadowPrefixQualification:
    def __init__(self):
        self.nav = self.codec = self.tap = None
        self.calls = 0

    def __call__(self, nav, observation):
        import numpy as np

        from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
        from scripts.rsi_motion_student_rollout import proprioception
        from scripts.rsi_motion_teacher_tap import MotionTeacherTap
        from scripts.rsi_motor_student_rollout import exact_array

        if self.calls == 0:
            self.nav = nav
            self.codec = FoundationMotorCodec(nav.backend)
            self.tap = MotionTeacherTap(nav.backend)
        if nav is not self.nav or observation.frame != self.calls or self.calls >= 20:
            raise ValueError("one fresh consecutive original20 shadow prefix required")
        self.codec.observe(
            np.asarray(observation.qpos),
            np.asarray(observation.qvel),
            self.codec._previous.copy(),
            frame=observation.frame,
        )
        advice = nav.prepare_shadow_action(observation)
        label = self.tap.label(proprioception(self.codec))
        exact_array(advice, label["teacher_advice_action"])
        target = nav.commit_batched_action(advice)
        self.codec.commit_consumed(advice, foundation_prefix=True)
        self.calls += 1
        if self.calls == 20:
            # The full forecast shares only original stateless ORT sessions.
            # Stop instrumentation before cloning; counters are not shared state.
            nav.backend._encoder = self.tap.encoder.session
            nav.backend._decoder = self.tap.decoder.session
        return target

    def verify_closed(self):
        if (
            self.calls != 20
            or self.codec._next_frame != 20
            or self.nav._next_frame < 20
            or self.tap.encoder.calls != 20
            or self.tap.decoder.calls != 20
        ):
            raise ValueError("all20 shadow prefix histories and full native endpoint required")
        return dict(
            shadow_encoder_calls=self.tap.encoder.calls,
            shadow_decoder_calls=self.tap.decoder.calls,
            checked_own_proprioception_frames=self.calls,
            actual_consumed_history930_byte_exact=True,
        )
