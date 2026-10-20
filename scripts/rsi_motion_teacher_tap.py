"""Read-only ONNX math capture for privileged own-state motion teacher labels."""

import numpy as np


class NetworkTap:
    def __init__(self, session, *, input_width, output_width):
        self.session = session
        self.input_width, self.output_width = input_width, output_width
        self.calls = 0
        self.last_input = self.last_output = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, outputs, inputs, *args, **kwargs):
        if len(inputs) != 1:
            raise ValueError("one original ONNX tensor input required")
        x = next(iter(inputs.values()))
        if x.shape != (1, self.input_width) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("finite original teacher input required")
        result = self.session.run(outputs, inputs, *args, **kwargs)
        y = result[0]
        if y.shape != (1, self.output_width) or y.dtype != np.float32 or not np.isfinite(y).all():
            raise ValueError("finite original teacher output required")
        self.last_input, self.last_output = x.copy(), y.copy()
        self.calls += 1
        return result


class MotionTeacherTap:
    def __init__(self, backend):
        self.encoder = NetworkTap(backend._encoder, input_width=1247, output_width=64)
        self.decoder = NetworkTap(backend._decoder, input_width=994, output_width=29)
        backend._encoder, backend._decoder = self.encoder, self.decoder
        self.recorded = 0

    def label(self, expected_proprioception):
        from scripts.rsi_motor_student_rollout import exact_array

        if self.encoder.calls != self.recorded + 1 or self.decoder.calls != self.recorded + 1:
            raise ValueError("exactly one fresh encoder and decoder call per own-state label")
        token = self.encoder.last_output[0]
        if (
            not np.array_equal(token * 16, np.round(token * 16))
            or token.min() < -1
            or token.max() > 0.9375
        ):
            raise ValueError("original FSQ32 teacher symbol required")
        exact_array(self.decoder.last_input[0, :64], token)
        exact_array(self.decoder.last_input[0, 64:], expected_proprioception)
        self.recorded += 1
        return dict(
            teacher_motion_token=token.copy(),
            teacher_advice_action=self.decoder.last_output[0].copy(),
            teacher_encoder_input=self.encoder.last_input[0].copy(),
        )
