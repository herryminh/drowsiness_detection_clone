import os
import pygame
from .config import ALARM_FILES


class AlarmPlayer:
    def __init__(self, volume=1.0):
        pygame.mixer.init()
        self.volume = volume

        # Lưu các file âm thanh
        self.sounds = {}
        self._load_all()

    def _load_all(self):
        """Load toàn bộ âm thanh một lần"""
        for key, path in ALARM_FILES.items():
            path = str(path)
            if not os.path.isfile(path):
                print(f"[Alarm] Missing file: {path}")
                self.sounds[key] = None
                continue

            try:
                self.sounds[key] = path
                print(f"[Alarm] Loaded: {key} -> {path}")
            except Exception as e:
                print(f"[Alarm] Error loading sound {key}: {e}")
                self.sounds[key] = None

    def is_playing(self):
        return pygame.mixer.music.get_busy()

    def play(self, alarm_type):
        """Phát đúng loại âm thanh"""
        if alarm_type not in self.sounds:
            print(f"[Alarm] Unknown alarm: {alarm_type}")
            return

        path = self.sounds[alarm_type]
        if path is None:
            print(f"[Alarm] Cannot play: file missing for {alarm_type}")
            return

        if not self.is_playing():  # Không chồng âm
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play(loops=0)
            except Exception as e:
                print(f"[Alarm] Failed to play {alarm_type}: {e}")

    def stop(self):
        """Theo yêu cầu: KHÔNG stop giữa chừng."""
        pass
