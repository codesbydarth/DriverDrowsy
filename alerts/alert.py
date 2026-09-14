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
                self._start_worker()

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
            self._mixer_initialized = True
            logger.info("Pygame audio mixer initialized at %d Hz.", self._sample_rate)
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

        try:
            for state, tone_specs in self.audio_config.synthetic_tones.items():
                audio_segments = []
                for freq, duration in tone_specs:
                    total_samples = int(self._sample_rate * duration)
                    t = np.linspace(0, duration, total_samples, endpoint=False)
                    # Smooth envelope to eliminate start/stop clicks
                    envelope = np.sin(np.pi * np.linspace(0, 1, total_samples)) ** 0.5
                    wave = np.sin(2 * np.pi * freq * t) * envelope * 32767
                    audio_segments.append(wave.astype(np.int16))

                    # 50ms pause between multi-tone pulses
                    gap_samples = int(self._sample_rate * 0.05)
                    audio_segments.append(np.zeros(gap_samples, dtype=np.int16))

                composite_wave = np.concatenate(audio_segments)
                sound = pygame.sndarray.make_sound(composite_wave)
                sound.set_volume(self.audio_config.volume)
                self._sounds[state] = sound

            logger.info("Synthesized %d alert audio tones.", len(self._sounds))
        except Exception as err:
            logger.warning("Failed to synthesize audio alert tones: %s", err)

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

        if self._mixer_initialized and pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        logger.info("AlertManager cleanly shut down.")
