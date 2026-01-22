import pyaudio
import speech_recognition as sr
import webrtcvad
#https://github.com/Shaileshsaravanan/realtime-python-captions/blob/main/main.py


# Initialize PyAudio for audio capture
audio_stream = pyaudio.PyAudio()
stream = audio_stream.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

# Initialize SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Initialize WebRTC Voice Activity Detection (VAD)
vad = webrtcvad.Vad()

# Set aggressiveness level (0 to 3) for VAD, adjust as needed
vad.set_mode(3)  # Aggressiveness level: 0 (least aggressive) to 3 (most aggressive)

# Main loop
try:
        # Read audio data from the audio stream
        audio_data = stream.read(1024)

        # Perform voice activity detection
        is_speech = vad.is_speech(audio_data, sample_rate=16000)

        if is_speech:
            # Convert audio data to AudioData object
            audio_data = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)

            # Perform speech recognition
            try:
                text = recognizer.recognize_google(audio_data)
                print(f"Captions: {text}")
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError:
                print("Speech-to-text service is down")

except KeyboardInterrupt:
    # Handle Ctrl+C to exit the loop
    pass
finally:
    # Release the audio stream
    stream.stop_stream()
    stream.close()
    audio_stream.terminate()