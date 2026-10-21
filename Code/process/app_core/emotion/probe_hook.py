"""The emotion probe's side of hidden-state capture: the format the patched native library emits, and the observer that
turns its samples into probe captures. inference/ knows neither: the factory hands a ProbeHook to the provider's ProbeHost
(attach_probe), which checks the library's /props against feature_version, calls start once the model is up and shows the
hook every generation (inference/provider.py: CaptureHook, GenerationObserver)."""
import math

# Also in tools/llama_cpp/emotion-probe.patch, which reports it in /props and in every sample (tests/test_release_build.py).
FEATURE_VERSION = 'result_norm-pool256-v1'
FEATURE_WIDTH = 256  # the patch average-pools result_norm to this many values: the input of probe.build_network
SAMPLE_EVENT = 'riko.emotion_probe.sample'
CAPTURE_SLOT = 0  # the patch arms the capture on slot 0 only: the live lane, and example replay at background priority


class ProbeHook:
    feature_version = FEATURE_VERSION

    def __init__(self, make_probe):
        self.make_probe, self.probe = make_probe, None  # make_probe(identity, idle) builds the EmotionProbe

    def start(self, identity, idle):
        """Build the probe for this backbone (identity names it, InProcessLlamaProvider._initialize_probe); idle says
        whether it may train now (ProbeHost.probe_idle)."""
        self.probe = self.make_probe(identity, idle)
        return self.probe

    def on_start(self, generation):
        if self.probe is not None and generation.slot == CAPTURE_SLOT: self.probe.activate(generation.group)

    def on_event(self, generation, event, visible):
        probe = self.probe
        if probe is None or generation.slot != CAPTURE_SLOT or event.get('type') != SAMPLE_EVENT: return
        features = event.get('features')
        # A sample belongs to the visible reply only when it was taken right after exactly this text (UTF-8 bytes).
        if not (event.get('feature_version') == FEATURE_VERSION and visible and event.get('prefix_bytes') == len(visible.encode('utf-8'))
                and isinstance(features, list) and len(features) == FEATURE_WIDTH
                and all(type(n) in (int, float) and math.isfinite(n) for n in features)): return
        import torch
        from ..kernel.output_filter import clean_output
        shown = clean_output(visible)
        last_user = next((m.content for m in reversed(generation.messages) if m.role == 'user'), '')
        group = generation.group
        probe.capture(torch.tensor(features, device='cpu'), f'user: {last_user}\nassistant: {shown}', group,
            cancelled=lambda: generation.cancelled() or probe.active_group != group,
            replay=generation.role == 'probe_replay', input_text=last_user, offset=len(shown))

    def on_finish(self, generation): pass

    def close(self):
        if self.probe is not None: self.probe.close()
