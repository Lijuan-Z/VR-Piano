from flask import Flask, request
import pygame
import threading

app = Flask(__name__)

# Initialize pygame mixer
pygame.mixer.init()

# Function to play sound asynchronously using pygame
def play_sound_async(sound_file):
    try:
        pygame.mixer.music.load(sound_file)  # Load the sound file
        pygame.mixer.music.play()  # Play the sound
    except Exception as e:
        print(f"Error playing sound: {e}")

@app.route('/play', methods=['POST'])
def play_sound():
    key = request.args.get('key')

    sound_files = {
        'LD': './sound/key05.mp3',
        'LE': './sound/key06.mp3',
        'LF': './sound/key07.mp3',
        'LG': './sound/key08.mp3',
        'A': './sound/key09.mp3',
        'B': './sound/key10.mp3',
        'C': './sound/key11.mp3',
        'D': './sound/key12.mp3',
        'E': './sound/key13.mp3',
        'F': './sound/key14.mp3',
        'G': './sound/key15.mp3',
        'RA': './sound/key16.mp3',
        'RB': './sound/key17.mp3',
        'RC': './sound/key18.mp3',
    }

    if key in sound_files:
        sound_file = sound_files[key]
        threading.Thread(target=play_sound_async, args=(sound_file,)).start()
        return f"Playing sound for key {key}", 200
    else:
        return f"Invalid key: {key}", 400

if __name__ == '__main__':
    app.run(debug=True)
