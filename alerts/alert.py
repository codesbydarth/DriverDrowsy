"""Non-blocking audio alert module using pygame.

Provides threaded, non-blocking acoustic warnings for driver fatigue states.
Synthesizes distinct in-memory audio waveforms (chirps, pulsing beeps, alarms)
without requiring external sound files. Includes cooldown management to prevent
audio stutter or queue saturation.
"""

import queue
import threading
import time
from typing import Dict, Optional
import numpy as np
import pygame

from utils.config import DEFAULT_CONFIG, AppConfig, ProjectState
from utils.logger import setup_logger

logger = setup_logger("alerts")


def adapt_waveform_channels(waveform: np.ndarray, target_channels: int) -> np.ndarray:
    """Adapt a 1D mono audio waveform to match the target mixer channel count.

    Args:
        waveform: 1D NumPy array representing mono audio signal.
        target_channels: Negotiated mixer channel count (1 for mono, 2 for stereo, >2 multi-channel).

    Returns:
        1D array if target_channels <= 1, or 2D array of shape (N, target_channels) where
        the mono signal is duplicated across all channels.
    """
    if target_channels <= 1 or waveform.ndim > 1:
        return waveform
    if target_channels == 2:
        return np.column_stack((waveform, waveform))
    return np.column_stack([waveform] * target_channels)


class AlertManager:
    """Manages non-blocking audio alerts with state-specific sound signatures."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize audio alert system and background playback thread.

        Args:
            config: Master application configuration instance.
        """
        self.config = config or DEFAULT_CONFIG
        self.audio_config = self.config.audio

        self._enabled = self.audio_config.enabled
        self._last_alert_time: float = 0.0
        self._last_alert_state: Optional[ProjectState] = None
        self._cooldown = self.audio_config.cooldown_seconds
        self._sample_rate = self.audio_config.sample_rate

        self._audio_queue: queue.Queue = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._sounds: Dict[ProjectState, pygame.mixer.Sound] = {}
        self._mixer_initialized = False

        if self._enabled:
            self._initialize_mixer()
            if self._mixer_initialized:
                self._synthesize_alert_tones()
                if self._sounds:
                    self._start_worker()
                else:
                    logger.warning("No audio alert sounds available. Disabling audio alerts.")
                    self._mixer_initialized = False
                    self._enabled = False

    def _initialize_mixer(self) -> None:
        """Initialize pygame mixer subsystem with graceful fallback on failure."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=self._sample_rate,
                    size=-16,
                    channels=1,
                    buffer=512
                )
            init_spec = pygame.mixer.get_init()
            if not init_spec:
                raise RuntimeError("pygame.mixer.get_init() returned None after init.")
            self._mixer_initialized = True
            logger.info(
                "Pygame audio mixer initialized: %d Hz, %d-bit format, %d channel(s).",
                init_spec[0], abs(init_spec[1]), init_spec[2]
            )
        except Exception as err:
            logger.warning(
                "Audio device unavailable or failed to initialize (%s). "
                "Disabling audio alerts.", err
            )
            self._mixer_initialized = False
            self._enabled = False

    def _synthesize_alert_tones(self) -> None:
        """Generate in-memory audio waveforms for each fatigue state."""
        if not self._mixer_initialized:
            return

        init_spec = pygame.mixer.get_init()
        if not init_spec:
            logger.warning("Pygame audio mixer not initialized; cannot synthesize tones.")
            self._mixer_initialized = False
            self._enabled = False
            return

        sample_rate, _, mixer_channels = init_spec

        try:
            for state, tone_specs in self.audio_config.synthetic_tones.items():
                audio_segments = []
                for freq, duration in tone_specs:
                    total_samples = int(sample_rate * duration)
                    t = np.linspace(0, duration, total_samples, endpoint=False)
                    # Smooth envelope to eliminate start/stop clicks
                    envelope = np.sin(np.pi * np.linspace(0, 1, total_samples)) ** 0.5
                    wave = np.sin(2 * np.pi * freq * t) * envelope * 32767
                    audio_segments.append(wave.astype(np.int16))

                    # 50ms pause between multi-tone pulses
                    gap_samples = int(sample_rate * 0.05)
                    audio_segments.append(np.zeros(gap_samples, dtype=np.int16))

                composite_wave = np.concatenate(audio_segments)
                adapted_wave = adapt_waveform_channels(composite_wave, mixer_channels)
                sound = pygame.sndarray.make_sound(adapted_wave)
                sound.set_volume(self.audio_config.volume)
                self._sounds[state] = sound

            logger.info(
                "Synthesized %d alert audio tones (%d channel(s) at %d Hz).",
                len(self._sounds), mixer_channels, sample_rate
            )
        except Exception as err:
            logger.warning("Failed to synthesize audio alert tones: %s", err)
            self._sounds.clear()
            self._mixer_initialized = False
            self._enabled = False

    def _start_worker(self) -> None:
        """Start the background playback daemon thread."""
        self._worker_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="AudioAlertWorker"
        )
        self._worker_thread.start()

    def _playback_worker(self) -> None:
        """Worker loop executing sound playback in a background thread."""
        while not self._stop_event.is_set():
            try:
                state = self._audio_queue.get(timeout=0.2)
                if state is None:
                    break

                sound = self._sounds.get(state)
                if sound is not None and self._mixer_initialized:
                    sound.play()
                self._audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as err:
                logger.error("Error during sound playback: %s", err)

    def trigger(self, state: ProjectState) -> None:
        """Trigger an audio alert for the given state if cooldown has elapsed.

        Non-blocking: puts the alert request on the background worker queue.

        Args:
            state: Current ProjectState to alert on.
        """
        if not self._enabled or not self._mixer_initialized:
            return

        # ALERT state requires no alarm
        if state == ProjectState.ALERT:
            return

        current_time = time.time()
        elapsed = current_time - self._last_alert_time

        # If entering a more severe state (e.g. DROWSY -> MICROSLEEP), bypass cooldown
        severity_escalation = (
            self._last_alert_state is not None and state > self._last_alert_state
        )

        if elapsed >= self._cooldown or severity_escalation:
            self._last_alert_time = current_time
            self._last_alert_state = state

            try:
                # Drop older queue items if full to prevent stale sounds
                if self._audio_queue.full():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._audio_queue.put_nowait(state)
            except Exception as err:
                logger.debug("Audio queue put error: %s", err)

    def close(self) -> None:
        """Cleanly stop playback worker and terminate mixer."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            try:
                self._audio_queue.put_nowait(None)
            except Exception:
                pass
            self._worker_thread.join(timeout=1.0)

        if pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        self._mixer_initialized = False
        logger.info("AlertManager cleanly shut down.")
