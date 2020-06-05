import tkinter
from tkinter.filedialog import askopenfilename
import os
import queue
import sys
import threading
import math
import time
import pyaudio
import wave

from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle

import librosa
import sounddevice as sd
import soundfile as sf

RESOLUTION="300x250"
REC_BUTTON_CONFIG = {
    "height": 2,
    "width": 15,
    "fg": "white",
    "bg": "red"
}
BUTTON_CONFIG = {
    "height": 2,
    "width": 15
}
REG_BUTTON_CONFIG = {
    "height": 2,
    "width": 15,
    "fg": "white",
    "bg": "green"
}
LABEL_CONFIG = {
    'wraplength': 500
}
FORMAT = pyaudio.paInt16

SAMPLE_RATE = 22050
CHANNELS = 2
FRAME_PER_BUFFER = 1024

RECORDING_FILE = "temp_record.wav"

q = queue.Queue()

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix

def clustering(X, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    return kmeans

def predict(file_path, is_delete):
    mfcc = get_mfcc(file_path)
    if is_delete:
        os.remove(file_path)

    # kmeans = clustering(mfcc)
    # mfcc = kmeans.predict(mfcc).reshape(-1,1)
    scores = {cname: model.score(mfcc, [len(mfcc)]) for cname, model in models.items()}
    cmax = max(scores.keys(), key=(lambda k: scores[k]))
    print(scores)
    print ("Predict: ", cmax)
    return cmax

class Recorder:
    def __init__(self):
        self.open_file_button = tkinter.Button(
            top,
            text="Open file",
            command=self.import_and_predict,
            **BUTTON_CONFIG
        )
        self.open_file_button.pack()
        self.open_file_look = False

        self.status = tkinter.Label(
            top,
            text=""
        )
        self.status.pack()
        self.status = tkinter.Label(
            top,
            text=""
        )
        self.status.pack()
        self.status = tkinter.Label(
            top,
            text=""
        )
        self.status.pack()

        self.start_button = tkinter.Button(
            top,
            text="REC",
            command=self.start_recording,
            **REC_BUTTON_CONFIG,
        )
        self.start_button.pack()
        self.start_lock = False

        self.stop_button = tkinter.Button(
            top,
            text="Stop REC",
            command=self.stop_recording,
            **BUTTON_CONFIG
        )
        self.stop_button.pack()
        self.stop_lock = True

        
        self.status = tkinter.Label(
            top,
            text=""
        )
        self.status.pack()
        self.status = tkinter.Label(
            top,
            text=""
        )
        self.status.pack()
        self.status = tkinter.Label(
            top,
            text=""
        )
        self.status.pack()
        self.status = tkinter.Label(
            top,
            text="Please import a wav file or record one!"
        )
        self.status.pack()

        self.recognize_button = tkinter.Button(
            top,
            text="Recognize",
            command=lambda: self.recognize(None, True),
            **REG_BUTTON_CONFIG
        )
        self.recognize_button.pack()
        self.recognize_lock = True

        self.is_recording = False

    def import_and_predict(self):
        file_path = askopenfilename()
        print(file_path)
        self.recognize(file_path, False)
        self.open_file_look = False

    def start_recording(self):
        if self.start_lock:
            return

        self.start_lock = True

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=FRAME_PER_BUFFER,
            input=True
        )

        self.frames = []

        self.is_recording = True
        self.status.config(text="Recording")

        self.recognize_lock = True
        self.stop_lock = False

        thread = threading.Thread(target=self.record)
        thread.start()

    def stop_recording(self):
        if self.stop_lock:
            return

        self.stop_lock = True

        self.is_recording = False

        wave_file = wave.open(RECORDING_FILE, "wb")

        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(self.audio.get_sample_size(FORMAT))
        wave_file.setframerate(SAMPLE_RATE)

        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        self.status.config(text="Recorded")

        self.recognize_lock = False
        self.start_lock = False

    def record(self):
        while (self.is_recording):
            data = self.stream.read(FRAME_PER_BUFFER)
            self.frames.append(data)

    def recognize(self, file_path, is_delete):
        if file_path == None:
            result = predict(os.path.join(RECORDING_FILE), is_delete)
            self.status.config(text=f"Predict: \"{result}\"")
        else:
            result = predict(file_path, is_delete)
            self.status.config(text=f"Predict: \"{result}\"")


def main():
    # ----------------------- tkinter config --------------------
    if os.environ.get('DISPLAY', '') == '':
        print('no display found. Using :0.0')
        os.environ.__setitem__('DISPLAY', ':0.0')

    global top
    top = tkinter.Tk()
    top.title("HMM Recognition")
    top.geometry(RESOLUTION)
    app = Recorder()
    # --------------------------------
    # Load models
    global models
    models = pickle.load(open("m.pkl", "rb"))
    class_names = ["toi", "theo", "dich", "nguoi", "benh_nhan"]
    top.mainloop()
    # -----------------------------------------------------------


    # pathfile = None
    # openButton = tkinter.Button(top, text="Open", fg='white', bg='grey', command=importFileAndPredict).place(bordermode='inside', x=0, y=0)
    # recButton = tkinter.Button(top, text="Rec", fg='white', bg='red', 
    #                 command=lambda: record("live_recording.wav")).place(x=125,y=25)
    # stopButton = tkinter.Button(top, text="Stop", fg='white', bg='grey', command=lambda: stop).place(x=200, y=25)
    # pathfile = None
    # P = tkinter.Button(top, text="Predict", fg='white', bg='green', command=lambda: predict(pathfile)).place(x=115, y=200)

    # inputThread = threading.Thread(target=read_kb_input, args=(inputQueue,), daemon=True)
    # inputThread.start()

    # quitButton = tkinter.Button(top, text="Quit", fg='white', bg='grey', command=top.destroy).place(bordermode='inside', x=240, y=0)
    # top.mainloop()
main()
