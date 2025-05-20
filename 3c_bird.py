import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import os
import datetime
import glob
import pygame
import time
import librosa
from itertools import groupby
import librosa.display
from scipy import stats
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Settings
sample_rate = 44100
default_duration = 15
output_dir = "3cwavs"

def ensure_output_directory():
    """Ensure the output directory exists"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

def record_audio(duration=default_duration):
    """Record audio for the specified duration and return the numpy array"""
    print(f"Recording for {duration} seconds... Speak or let the bird chirp")
    print("Press Ctrl+C to stop recording early")
    
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        print("Recording finished!")
        return audio
    except KeyboardInterrupt:
        sd.stop()
        print("\nRecording stopped early.")
        return None
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def save_recording(audio, sample_rate):
    """Save the recording with a timestamp filename"""
    ensure_output_directory()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/recording_{timestamp}.wav"
    
    try:
        write(filename, sample_rate, audio)
        print(f"Recording saved as: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving recording: {e}")
        return None

def visualize_audio(audio, duration):
    """Plot the audio waveform"""
    try:
        # Flatten array for plotting
        audio_flat = audio.flatten()
        
        # Plot waveform
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, duration, len(audio_flat)), audio_flat)
        plt.title("Audio Waveform")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing audio: {e}")

def list_recordings():
    """List all available WAV recordings"""
    ensure_output_directory()
    recordings = glob.glob(f"{output_dir}/*.wav")
    
    if not recordings:
        print("No recordings found.")
        return []
    
    print("\nAvailable recordings:")
    for i, recording in enumerate(recordings, 1):
        # Get file size in MB
        size_mb = os.path.getsize(recording) / (1024 * 1024)
        # Get creation time
        creation_time = datetime.datetime.fromtimestamp(os.path.getctime(recording))
        creation_time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{i}. {os.path.basename(recording)} - {size_mb:.2f} MB - {creation_time_str}")
    
    return recordings

def play_recording(filename):
    """Play back a WAV recording"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        print(f"Playing: {os.path.basename(filename)}")
        print("Press Ctrl+C to stop playback")
        
        pygame.mixer.music.play()
        # Wait for playback to finish or for user to interrupt
        try:
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pygame.mixer.music.stop()
            print("\nPlayback stopped.")
        
        pygame.mixer.quit()
    except Exception as e:
        print(f"Error playing recording: {e}")

def analyze_pitch_map(filename):
    """Analyze and visualize sound pitches in a recording"""
    try:
        print(f"Analyzing pitches in: {os.path.basename(filename)}")
        # Load audio file with librosa
        y, sr = librosa.load(filename)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get pitch track
        pitch_track = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:
                pitch_track.append(pitch)
        
        # Define pitch ranges and assign labels
        def pitch_to_symbol(freq):
            if freq < 1000:
                return "L"  # Low
            elif 1000 <= freq < 2000:
                return "A"
            elif 2000 <= freq < 3000:
                return "B"
            elif 3000 <= freq < 4000:
                return "C"
            elif 4000 <= freq < 5000:
                return "D"
            else:
                return "H"  # High
        
        # Map to symbols
        symbol_sequence = [pitch_to_symbol(f) for f in pitch_track]
        
        # Collapse repeated adjacent symbols
        compressed_sequence = [k for k, _ in groupby(symbol_sequence)]
        
        # Show the results
        print("Raw Symbol Sequence:")
        print(symbol_sequence[:100] + ['...'] if len(symbol_sequence) > 100 else symbol_sequence)
        
        print("\nCompressed Pattern (removes immediate repeats):")
        print(compressed_sequence[:100] + ['...'] if len(compressed_sequence) > 100 else compressed_sequence)
        
        # Create the symbol pattern visualization
        analyze_sound_pattern(compressed_sequence)
        
        # Also show pitch distribution
        plt.figure(figsize=(12, 4))
        plt.hist(pitch_track, bins=50, alpha=0.7)
        plt.title("Pitch Distribution")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing pitches: {e}")

def advanced_analysis(filename):
    """Perform advanced analysis on audio recording with natural language descriptions"""
    try:
        print(f"\nPerforming advanced analysis on: {os.path.basename(filename)}")
        # Load audio file with librosa
        y, sr = librosa.load(filename)
        
        # Run all analyses
        pitch_analysis_results = analyze_pitch_advanced(y, sr)
        rhythm_analysis_results = analyze_rhythm(y, sr)
        pattern_description = generate_pattern_description(y, sr, pitch_analysis_results)
        
        # Display comprehensive visualizations
        enhanced_visualizations(y, sr, 
                               pitch_analysis_results,
                               rhythm_analysis_results)
        
        # Pattern matching against known bird species
        match_bird_patterns(pitch_analysis_results, rhythm_analysis_results)
        
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_pitch_advanced(y, sr):
    """Enhanced pitch analysis with statistics and pattern recognition"""
    results = {}
    
    # Extract pitch information
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get pitch track (non-zero pitches)
    pitch_track = []
    pitch_times = []
    
    print("\n📊 Frequency Analysis:")
    
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_track.append(pitch)
            pitch_times.append(i * 512 / sr)  # Convert frame to time
    
    # Calculate statistics
    if pitch_track:
        min_freq = min(pitch_track)
        max_freq = max(pitch_track)
        avg_freq = sum(pitch_track) / len(pitch_track)
        
        # Calculate dominant frequency using numpy's unique function
        try:
            # Round frequencies to nearest 10 Hz for better grouping
            rounded_pitches = np.round(pitch_track, -1)
            # Get unique values and their counts
            unique_freqs, counts = np.unique(rounded_pitches, return_counts=True)
            # Find the index of the most frequent value
            idx_max = np.argmax(counts)
            # Get the most frequent value
            dominant_freq = unique_freqs[idx_max]
        except Exception as e:
            print(f"  • Warning: Could not determine dominant frequency: {e}")
            dominant_freq = avg_freq  # Fallback to average if mode calculation fails
        
        print(f"  • Frequency Range: {min_freq:.2f}Hz - {max_freq:.2f}Hz")
        print(f"  • Average Frequency: {avg_freq:.2f}Hz")
        print(f"  • Dominant Frequency: {dominant_freq:.2f}Hz")
        
        # Calculate pitch intervals (differences between consecutive pitches)
        intervals = [pitch_track[i+1] - pitch_track[i] for i in range(len(pitch_track)-1)]
        
        # Melodic contour analysis
        contour_changes = []
        for i in range(len(intervals)):
            if intervals[i] > 10:  # Rising by more than 10Hz
                contour_changes.append('↗')
            elif intervals[i] < -10:  # Falling by more than 10Hz
                contour_changes.append('↘')
            else:  # Relatively stable
                contour_changes.append('→')
        
        # Count contour patterns
        rising_count = contour_changes.count('↗')
        falling_count = contour_changes.count('↘')
        stable_count = contour_changes.count('→')
        
        total_changes = len(contour_changes)
        if total_changes > 0:
            print("\n📈 Melodic Contour Analysis:")
            print(f"  • Rising: {rising_count} ({rising_count/total_changes*100:.1f}%)")
            print(f"  • Falling: {falling_count} ({falling_count/total_changes*100:.1f}%)")
            print(f"  • Stable: {stable_count} ({stable_count/total_changes*100:.1f}%)")
            
            # Identify trill patterns (rapid alternation between two pitches)
            trill_pattern = ['↗', '↘', '↗', '↘']
            trill_count = 0
            for i in range(len(contour_changes) - 3):
                if contour_changes[i:i+4] == trill_pattern:
                    trill_count += 1
            
            if trill_count > 0:
                print(f"  • Detected {trill_count} trill patterns")
        
        # Store results
        results['pitch_track'] = pitch_track
        results['pitch_times'] = pitch_times
        results['min_freq'] = min_freq
        results['max_freq'] = max_freq
        results['avg_freq'] = avg_freq
        results['intervals'] = intervals
        results['contour_changes'] = contour_changes
        results['trill_count'] = trill_count if 'trill_count' in locals() else 0
    else:
        print("  • No clear pitches detected")
        
    return results

def analyze_rhythm(y, sr):
    """Analyze rhythmic features of the audio"""
    results = {}
    
    print("\n🥁 Rhythm Analysis:")
    
    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    
    
    print(f"  • Onset strength mean: {np.mean(onset_env):.4f}, max: {np.max(onset_env):.4f}")
    
    # Beat detection
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=120, units='time')
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo = float(tempo)
    
    print("  • Onset Envelope Mean:", np.mean(onset_env))
    print("  • Max Onset Strength:", np.max(onset_env))
    
    print(f"  • Estimated Tempo: {tempo:.1f} BPM")
    print(f"  • Detected {len(onsets)} onsets and {len(beats)} beats")
    
    # Calculate inter-onset intervals
    if len(onset_times) > 1:
        ioi = np.diff(onset_times)
        avg_ioi = np.mean(ioi)
        std_ioi = np.std(ioi)
        
        print(f"  • Average time between onsets: {avg_ioi*1000:.1f}ms")
        
        # Check rhythmic regularity
        if std_ioi < 0.05:  # Low standard deviation indicates regular rhythm
            print("  • Rhythm pattern: Very regular (mechanical/constant)")
        elif std_ioi < 0.1:
            print("  • Rhythm pattern: Mostly regular")
        else:
            print("  • Rhythm pattern: Irregular/varied")
        
        # Find repeated rhythmic patterns
        rhythm_patterns = []
        for i in range(len(ioi) - 2):
            pattern = (ioi[i], ioi[i+1])
            rhythm_patterns.append(pattern)
        
        # Count pattern occurrences
        pattern_counts = Counter([str(p) for p in rhythm_patterns])
        most_common = pattern_counts.most_common(1)
        
        if most_common and most_common[0][1] > 2:  # Pattern occurs more than twice
            print(f"  • Detected a recurring rhythmic pattern ({most_common[0][1]} occurrences)")
        
        results['tempo'] = tempo
        results['onset_times'] = onset_times
        results['beat_times'] = beat_times
        results['ioi'] = ioi
        results['avg_ioi'] = avg_ioi
        results['std_ioi'] = std_ioi
    else:
        print("  • Not enough onsets to analyze rhythm patterns")
    
    return results

def generate_pattern_description(y, sr, pitch_results):
    """Generate natural language description of sound patterns"""
    description = []
    print("\n🔊 Sound Pattern Description:")
    
    # Get basic audio characteristics
    duration = librosa.get_duration(y=y, sr=sr)
    pitch_track = pitch_results.get('pitch_track', [])
    contour_changes = pitch_results.get('contour_changes', [])
    
    # Describe overall audio
    description.append(f"This is a {duration:.1f}-second audio recording.")
    
    # Describe frequency characteristics if we have pitch data
    if pitch_track:
        # Classify pitch range
        min_freq = pitch_results['min_freq']
        max_freq = pitch_results['max_freq']
        
        if max_freq < 1000:
            range_desc = "low-frequency"
        elif min_freq > 4000:
            range_desc = "high-frequency"
        elif 1000 <= min_freq <= 4000 and 1000 <= max_freq <= 4000:
            range_desc = "mid-frequency"
        else:
            range_desc = "mixed-frequency"
        
        freq_range = max_freq - min_freq
        if freq_range < 200:
            range_size = "narrow"
        elif freq_range < 1000:
            range_size = "moderate"
        else:
            range_size = "wide"
        
        description.append(f"It contains primarily {range_desc} sounds with a {range_size} pitch range.")
        
        # Analyze pitch contour
        if contour_changes:
            rising = contour_changes.count('↗')
            falling = contour_changes.count('↘')
            stable = contour_changes.count('→')
            total = len(contour_changes)
            
            # Determine contour type
            if rising > falling and rising > stable:
                contour = "primarily ascending (rising pitch)"
            elif falling > rising and falling > stable:
                contour = "primarily descending (falling pitch)"
            elif stable > rising and stable > falling:
                contour = "mostly stable (consistent pitch)"
            elif abs(rising - falling) < 0.1 * total and rising + falling > 0.8 * total:
                contour = "highly varied with frequent changes"
                if pitch_results.get('trill_count', 0) > 2:
                    contour += " and multiple trills"
            else:
                contour = "mixed with some variation"
                
            description.append(f"The melody pattern is {contour}.")

            # Check for patterns that might match bird calls
            pattern_types = []
            
            # Look for rapid repeated patterns (common in many birds)
            trill_sections = 0
            for i in range(0, len(contour_changes) - 10, 5):
                section = contour_changes[i:i+10]
                if section.count('↗') > 3 and section.count('↘') > 3:
                    trill_sections += 1
                    
            if trill_sections > 0:
                pattern_types.append(f"{trill_sections} trill-like sections")
                
            # Look for whistles (long stable sections)
            whistle_count = 0
            stable_run = 0
            for c in contour_changes:
                if c == '→':
                    stable_run += 1
                else:
                    if stable_run > 5:  # Sustained note
                        whistle_count += 1
                    stable_run = 0
                    
            if stable_run > 5:  # Check the last run too
                whistle_count += 1
                
            if whistle_count > 0:
                pattern_types.append(f"{whistle_count} whistle-like sounds")
                
            # Describe pattern complexity
            if len(set(contour_changes)) < 2 and len(contour_changes) > 10:
                complexity = "simple and repetitive"
            elif len(set(''.join(contour_changes[i:i+3]) for i in range(len(contour_changes)-2))) < 5:
                complexity = "moderately complex"
            else:
                complexity = "complex and varied"
                
            description.append(f"The overall pattern is {complexity}.")
            
            # Add detected pattern types
            if pattern_types:
                description.append(f"Notable elements include {', '.join(pattern_types)}.")
    else:
        description.append("No clear pitch pattern could be detected.")
        
    # Print the natural language description
    for line in description:
        print(f"  • {line}")
        
    return description

def enhanced_visualizations(y, sr, pitch_results, rhythm_results):
    """Create enhanced visualizations for audio analysis"""
    # Create figure with multiple subplots
    plt.figure(figsize=(14, 10))
    
    # 1. Waveform and detected onsets/beats
    plt.subplot(3, 1, 1)
    plt.title('Waveform with Rhythm Analysis')
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    
    # Plot onsets if available
    if 'onset_times' in rhythm_results:
        onset_times = rhythm_results['onset_times']
        onset_heights = np.max(np.abs(y)) * 0.9
        plt.vlines(onset_times, -onset_heights, onset_heights, color='r', alpha=0.7, 
                   label=f'Onsets ({len(onset_times)})')
    
    # Plot beats if available
    if 'beat_times' in rhythm_results:
        beat_times = rhythm_results['beat_times']
        beat_heights = np.max(np.abs(y)) * 0.7
        plt.vlines(beat_times, -beat_heights, beat_heights, color='g', linestyle='--', alpha=0.7,
                  label=f'Beats ({len(beat_times)})')
    
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # 2. Spectrogram with pitch tracking
    plt.subplot(3, 1, 2)
    plt.title('Spectrogram with Pitch Tracking')
    
    # Generate spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    
    # Overlay pitch track if available
    if 'pitch_track' in pitch_results and 'pitch_times' in pitch_results:
        pitch_track = pitch_results['pitch_track']
        pitch_times = pitch_results['pitch_times']
        if len(pitch_track) > 0 and len(pitch_track) == len(pitch_times):
            plt.plot(pitch_times, pitch_track, 'r.', alpha=0.5, label='Pitch Track')
            plt.legend(loc='upper right')
    
    # 3. Power spectrum analysis
    plt.subplot(3, 1, 3)
    plt.title('Frequency Power Spectrum')
    
    # FFT to get power spectrum
    n_fft = 2048
    ft = np.abs(librosa.stft(y, n_fft=n_fft))
    power_spectrum = np.mean(ft, axis=1)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Only show up to 8kHz (or less if the max frequency is less)
    max_freq = min(8000, sr/2)
    mask = freq_bins < max_freq
    plt.semilogy(freq_bins[mask], power_spectrum[mask], label='Power')
    
    # Mark dominant frequencies
    if 'pitch_track' in pitch_results and len(pitch_results['pitch_track']) > 0:
        # Get the 3 most common frequencies (rounded to nearest 10Hz)
        dominant_freqs = Counter(np.round(pitch_results['pitch_track'], -1)).most_common(3)
        for freq, count in dominant_freqs:
            # Find closest bin to this frequency
            idx = (np.abs(freq_bins - freq)).argmin()
            power = power_spectrum[idx]
            plt.plot(freq, power, 'ro', markersize=8)
            plt.annotate(f'{freq:.0f}Hz', 
                       xy=(freq, power),
                       xytext=(freq+50, power*1.2),
                       arrowprops=dict(arrowstyle="->", color='black'))
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True, which="both", ls="-", alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def match_bird_patterns(pitch_results, rhythm_results):
    """Match audio patterns against common bird call characteristics"""
    print("\n🐦 Bird Pattern Analysis:")
    
    # Define a simple database of common bird call patterns
    bird_patterns = {
        "European Robin": {
            "freq_range": (2000, 6000),
            "tempo_range": (80, 140),
            "pattern": "melodic with regular pauses, mostly stable pitch",
            "description": "Clear, flute-like warbling with pauses"
        },
        "Common Blackbird": {
            "freq_range": (1800, 4000),
            "tempo_range": (60, 110),
            "pattern": "varied phrases with some repetition, wide pitch range",
            "description": "Rich, flute-like singing with varied melodies"
        },
        "Great Tit": {
            "freq_range": (3000, 7000),
            "tempo_range": (90, 200),
            "pattern": "repeated identical notes, consistent rhythm",
            "description": "Clear 'teacher-teacher' call with two-note phrases"
        },
        "Eurasian Wren": {
            "freq_range": (2500, 8000),
            "tempo_range": (140, 280),
            "pattern": "rapid trills, complex patterns, wide frequency range",
            "description": "Explosive, complex song with rapid trilling"
        },
        "Common Chaffinch": {
            "freq_range": (2000, 5000),
            "tempo_range": (100, 180),
            "pattern": "accelerating rhythm ending with flourish",
            "description": "Short, energetic phrases ending with a distinctive flourish"
        },
        "European Goldfinch": {
            "freq_range": (2500, 6000),
            "tempo_range": (120, 200),
            "pattern": "rapid high-pitched trills with liquid quality",
            "description": "Tinkling, liquid call with rapid variations"
        },
        "Common Chiffchaff": {
            "freq_range": (3500, 6500),
            "tempo_range": (140, 220),
            "pattern": "simple alternating two-note pattern",
            "description": "Simple repeating 'chiff-chaff' pattern"
        },
        "Eurasian Blackcap": {
            "freq_range": (2000, 5500),
            "tempo_range": (100, 170),
            "pattern": "rich warbling with flute-like quality",
            "description": "Rich, varied flute-like warbling with pure tones"
        }
    }
    
    # Extract features from our analysis
    matches = []
    
    # If we have the necessary data, try to match patterns
    if pitch_results and 'min_freq' in pitch_results and 'max_freq' in pitch_results:
        min_freq = pitch_results['min_freq']
        max_freq = pitch_results['max_freq']
        
        # Get rhythm info if available
        tempo = rhythm_results.get('tempo', 0)
        
        # Get trill information if available
        has_trills = pitch_results.get('trill_count', 0) > 0
        
        # Calculate pattern features
        freq_range = max_freq - min_freq
        varied_pitch = freq_range > 1000  # If the pitch range is wide
        
        # Pattern matching logic
        for bird, pattern in bird_patterns.items():
            score = 0
            reasons = []
            
            # Check frequency range overlap
            bird_min, bird_max = pattern["freq_range"]
            if bird_min <= max_freq and bird_max >= min_freq:
                # Calculate overlap percentage
                overlap_min = max(bird_min, min_freq)
                overlap_max = min(bird_max, max_freq)
                overlap = (overlap_max - overlap_min) / (max_freq - min_freq)
                score += overlap * 0.4  # Weight: 40%
                reasons.append(f"Frequency range overlap: {overlap*100:.1f}%")
            
            # Check tempo if available
            if tempo > 0:
                bird_tempo_min, bird_tempo_max = pattern["tempo_range"]
                if bird_tempo_min <= tempo <= bird_tempo_max:
                    tempo_match = 1 - abs(tempo - (bird_tempo_min + bird_tempo_max)/2) / (bird_tempo_max - bird_tempo_min)
                    score += tempo_match * 0.2  # Weight: 20%
                    reasons.append(f"Tempo match: {tempo_match*100:.1f}%")
            
            # Check pattern description matches
            if "trill" in pattern["pattern"] and has_trills:
                score += 0.2
                reasons.append("Trill pattern detected (match)")
            
            if "varied" in pattern["pattern"] and varied_pitch:
                score += 0.1
                reasons.append("Varied pitch pattern (match)")
                
            if "repeated" in pattern["pattern"] and not varied_pitch:
                score += 0.1
                reasons.append("Repetitive pattern (match)")
                
            # Only consider scores above a threshold
            if score > 0.3:
                matches.append({
                    "bird": bird,
                    "score": score,
                    "reasons": reasons,
                    "description": pattern["description"]
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Report findings
        if matches:
            print("  • Potential bird species matches:")
            for i, match in enumerate(matches[:3], 1):  # Show top 3 matches
                print(f"    {i}. {match['bird']} - {match['score']*100:.1f}% confidence")
                print(f"       {match['description']}")
                print(f"       Matching features: {'; '.join(match['reasons'])}")
                print()
        else:
            print("  • No clear matches to known bird patterns in our database")
            print("  • This could be ambient sound, an uncommon bird, or non-bird audio")
    else:
        print("  • Insufficient data to perform bird pattern matching")
    
    return matches

def advanced_analysis_menu():
    """Menu for advanced audio analysis"""
    selected_file = file_selection_menu("analyze")
    
    if selected_file:
        advanced_analysis(selected_file)

def analyze_sound_pattern(compressed_sequence):
    """Visualize sound symbol pattern"""
    try:
        if isinstance(compressed_sequence, str):
            # If we're passed a filename instead of a sequence
            return analyze_pitch_map(compressed_sequence)
            
        # Assign each symbol a unique number for plotting
        symbol_to_num = {'L': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'H': 5}
        symbol_nums = [symbol_to_num[sym] for sym in compressed_sequence if sym in symbol_to_num]
        
        if not symbol_nums:
            print("No valid symbols found in the sequence")
            return
            
        plt.figure(figsize=(12, 4))
        plt.plot(symbol_nums, marker='o', linestyle='-', color='teal')
        plt.yticks(list(symbol_to_num.values()), list(symbol_to_num.keys()))
        plt.title("Symbolic Pattern Over Time")
        plt.xlabel("Sequence Step")
        plt.ylabel("Symbol")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing sound pattern: {e}")

def file_selection_menu(purpose="play"):
    """Generic file selection menu that returns the selected file path or None"""
    recordings = list_recordings()
    
    if not recordings:
        return None
    
    while True:
        try:
            choice = input(f"\nEnter recording number to {purpose} (or 'b' to go back): ")
            
            if choice.lower() == 'b':
                return None
            
            try:
                index = int(choice) - 1
                if 0 <= index < len(recordings):
                    return recordings[index]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number or 'b'.")
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            return None

def playback_menu():
    """Show the playback menu and handle user selection"""
    selected_file = file_selection_menu("play")
    
    if not selected_file:
        return
    
    play_recording(selected_file)
    
    # Ask if user wants to visualize this recording
    viz_choice = input("Visualize this recording? (y/n): ")
    if viz_choice.lower() == 'y':
        # Read the audio file and visualize it
        try:
            sample_rate, audio = read(selected_file)
            duration = len(audio) / sample_rate
            visualize_audio(audio, duration)
        except Exception as e:
            print(f"Error visualizing audio: {e}")

def analyze_pattern_menu():
    """Show menu for analyzing sound pattern"""
    selected_file = file_selection_menu("analyze")
    
    if selected_file:
        analyze_sound_pattern(selected_file)

def analyze_pitches_menu():
    """Show menu for analyzing pitch map"""
    selected_file = file_selection_menu("analyze")
    
    if selected_file:
        analyze_pitch_map(selected_file)

def main_menu():
    """Display the main menu and handle user selections"""
    while True:
        print("\n===== 3C Bird Audio Recorder =====")
        print("1. Record new audio")
        print("2. Play recordings")
        print("3. Analyze Sound Pattern")
        print("4. Map Sound Pitches")
        print("5. Advanced Analysis")
        print("6. Exit")
        
        try:
            choice = input("Enter your choice (1-6): ")
            
            if choice == '1':
                # Ask for recording duration
                try:
                    duration_input = input(f"Enter recording duration in seconds (default: {default_duration}): ")
                    duration = int(duration_input) if duration_input.strip() else default_duration
                except ValueError:
                    print(f"Invalid input. Using default duration of {default_duration} seconds.")
                    duration = default_duration
                
                # Record audio
                audio = record_audio(duration)
                if audio is not None:
                    filename = save_recording(audio, sample_rate)
                    if filename:
                        visualize_audio(audio, duration)
            
            elif choice == '2':
                playback_menu()
            
            elif choice == '3':
                analyze_pattern_menu()
                
            elif choice == '4':
                analyze_pitches_menu()
                
            elif choice == '5':
                advanced_analysis_menu()
                
            elif choice == '6':
                print("Exiting. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
        
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print(f"Fatal error: {e}")
