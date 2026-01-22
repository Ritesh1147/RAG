from gtts import gTTS
import os

import speech_recognition as sr
import pysrt


from pydub import AudioSegment # uses FFMPEG
import speech_recognition as sr

def process(filepath, chunksize=60000):
    #0: load mp3
    sound = AudioSegment.from_mp3(filepath)

    #1: split file into 60s chunks
    def divide_chunks(sound, chunksize):
        # looping till length l
        for i in range(0, len(sound), chunksize):
            yield sound[i:i + chunksize]
    chunks = list(divide_chunks(sound, chunksize))
    print(f"{len(chunks)} chunks of {chunksize/1000}s each")

    r = sr.Recognizer()
    #2: per chunk, save to wav, then read and run through recognize_google()
    string_index = {}
    for index,chunk in enumerate(chunks):
        #TODO io.BytesIO()
        temp = 'temp.wav'
        chunk.export(temp, format='wav')
        with sr.AudioFile(temp) as source:
            audio = r.record(source)
        #s = r.recognize_google(audio, language="en-US") #, key=API_KEY) --- my key results in broken pipe
        s = r.recognize_google(audio, language="en-US")
        string_index[index] = s
        break
    return ' '.join([string_index[i] for i in range(len(string_index))])





def audio_to_captions(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        # Format text into captions with timestamps (simplified)
        captions = pysrt.SubRipFile()
        captions.append(pysrt.SubRipItem(1, start=0, end=10, text=text))
        return captions
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from service; {e}"
'''
if __name__ == "__main__":
    audio_file = "audio.wav"  # Replace with your audio file
    captions = audio_to_captions(audio_file)
    if isinstance(captions, pysrt.SubRipFile):
        # Print or display captions using your custom method
        for caption in captions:
            print(caption.text)
    else:
        print(captions)

'''

def text_to_audio(text, language='en'):
    mp3_file = f'{language}_output.mp3'
    mp3_file="test.wav"
    
    tts_file = gTTS(text=text, lang=language, slow=False)
    tts_file.save(mp3_file)


   
        
#text = "Hello, this is a test."
#text_to_audio(text)

# import required module
import os

#audio_file = "test.wav"
audio_file = "static/Picture3.wav"

captions = audio_to_captions(audio_file)
#audio_file_name = 'audio.mp3'
#captions = process(audio_file)
print(captions)
audio_filemp3 = "static/Picture3.mp3"

# play sound

print('playing sound using native player')
os.system("mpg123 " + audio_filemp3)

if isinstance(captions, pysrt.SubRipFile):
    # Print or display captions using your custom method
   for caption in captions:
      print("Captions.text")
      print(caption.text)
   else:
      print(captions)