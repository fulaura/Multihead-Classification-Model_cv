import threading
import simpleaudio as sa

def play_sound():
    wave_obj = sa.WaveObject.from_wave_file("alert.wav")
    wave_obj.play()

threading.Thread(target=play_sound).start()
