import pygame
import keyboard

# Function to play audio
def play_audio():
    global is_playing
    pygame.mixer.music.play()
    is_playing = True

# Function to pause audio
def pause_audio():
    global is_playing
    pygame.mixer.music.pause()
    is_playing = False

# Function to resume audio
def resume_audio():
    global is_playing
    pygame.mixer.music.unpause()
    is_playing = True

# Function to toggle pause/resume on space key press
def pause_resume_audio():
    if is_playing:
        pause_audio()
        print("Paused")
    else:
        resume_audio()
        print("Resumed")

if __name__ == '__main__':
    # Initialize Pygame
    pygame.init()

    # Load the audio file
    audio_file_path = r"D:\STUDY\DHSP\NCKH-2023-With my idol\Vi6\output\output.mp3"
    pygame.mixer.music.load(audio_file_path)

    # Flag to track if audio is playing
    is_playing = False

    # Register space key press event
    keyboard.add_hotkey('space', lambda: pause_resume_audio())

    # Play the audio
    play_audio()

    # Keep the program running until the user presses the 'esc' key
    keyboard.wait('esc')

    # Quit Pygame
    pygame.quit()
