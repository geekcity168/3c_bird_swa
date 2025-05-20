import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import tempfile
import time

st.set_page_config(page_title="Bird Chirp Decoder", layout="centered")

st.title("🐦 Bird Chirp Decoder")
st.markdown("Record bird sounds and see pitch, rhythm, and symbolic patterns!")

# Record duration input
duration = st.slider("Recording Duration (seconds)", 3, 15, 5)

if st.button("🎙️ Record Now"):
    with st.spinner("Recording... chirp away!"):
        fs = 44100
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        st.success("Recording complete!")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            librosa.output.write_wav(tmp_path, audio.flatten(), sr=fs)

        # Load for analysis
        y, sr = librosa.load(tmp_path)
        st.audio(tmp_path, format='audio/wav')

        # Pitch Extraction
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_track = []
        for i in range(pitches.shape[1]):
            idx = magnitudes[:, i].argmax()
            pitch = pitches[idx, i]
            if pitch > 0:
                pitch_track.append(pitch)

        # Symbol Mapping
        def pitch_to_symbol(freq):
            if freq < 1000: return "L"
            elif freq < 2000: return "A"
            elif freq < 3000: return "B"
            elif freq < 4000: return "C"
            elif freq < 5000: return "D"
            else: return "H"

        symbols = [pitch_to_symbol(p) for p in pitch_track]
        from itertools import groupby
        compressed = [k for k, _ in groupby(symbols)]

        st.subheader("📊 Symbolic Pattern")
        st.write(' '.join(compressed[:50]) + ('...' if len(compressed) > 50 else ''))

        # Pitch Graph
        st.subheader("🎵 Pitch Pattern")
        fig, ax = plt.subplots()
        ax.plot(pitch_track, color='purple')
        ax.set_title("Pitch over Time")
        ax.set_ylabel("Frequency (Hz)")
        st.pyplot(fig)

        # Tempo (Rhythm)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=120)
        st.subheader("🥁 Estimated Tempo")
        st.write(f"{tempo:.1f} BPM")

        # Fun Translation (optional)
        st.subheader("🔊 Interpretation (fun)")
        if tempo > 150:
            st.write("🚨 Alert call! Bird might be warning others.")
        elif 'C' in compressed or 'H' in compressed:
            st.write("💬 Energetic or excited. Could be territorial.")
        else:
            st.write("🌿 Calm and steady — probably a regular chirp.")

