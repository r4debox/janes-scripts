#!/usr/bin/env python3
# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <https://unlicense.org/>


import sys
import os
import numpy as np
import librosa
import librosa.display
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm  # for highlighting negative values
import csv
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting functionality

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QLineEdit, QFileDialog, QMessageBox, QFormLayout, QCheckBox,
    QTabWidget, QGroupBox, QSlider
)
from PyQt5.QtCore import Qt

# Global variable to store the debug file path if provided
DEBUG_FILE = None
print(f"[CREATOR] janerain on discord")
print(f"[CREATOR] https://github.com/r4debox")
def debug_print(message):
    # Cute debug helper uwu: prints a debug message and logs it if needed.
    out_msg = f"[DEBUG] {message}"
    print(out_msg)
    if DEBUG_FILE is not None:
        try:
            with open(DEBUG_FILE, "a") as f:
                f.write(out_msg + "\n")
        except Exception as e:
            print(f"[DEBUG] Oops, couldn't write debug info: {e}")

def my_exception_hook(exctype, value, traceback_obj):
    # Critical exception handler that logs the error details and halts the program.
    import traceback
    err_msg = "Uncaught exception: " + str(value)
    print(err_msg)
    traceback.print_exception(exctype, value, traceback_obj)
    if DEBUG_FILE is not None:
        try:
            with open(DEBUG_FILE, "a") as f:
                f.write(err_msg + "\n")
                traceback.print_exception(exctype, value, traceback_obj, file=f)
        except Exception as e:
            print(f"[DEBUG] Failed to log exception: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

sys.excepthook = my_exception_hook

class AudioVisualizer(QMainWindow):
    def __init__(self):
        # Initialize the main window and set up initial state variables.
        super().__init__()
        self.setWindowTitle("Audio Visualizer and Analyzer")
        self.audio_file = None        # Currently loaded audio file path
        self.y = None                 # Audio time series
        self.sr = 44100               # Default sample rate
        self.current_spec_data = None # Data for main visualization
        self.extra_plots_data = []    # Additional features data (list of tuples)
        self.entropy_data = None      # Computed entropy values
        self.entropy_times = None     # Corresponding times for entropy plot
        debug_print("Initializing AudioVisualizer GUI uwu~")
        self.initUI()

    def initUI(self):
        # Set up the GUI layout with tabs for visualization and advanced settings.
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Visualization Tab
        visWidget = QWidget()
        visLayout = QVBoxLayout()
        visWidget.setLayout(visLayout)

        topLayout = QHBoxLayout()
        self.fileLabel = QLabel("No file selected")
        loadBtn = QPushButton("Load Audio File")
        loadBtn.clicked.connect(self.loadFile)
        topLayout.addWidget(self.fileLabel)
        topLayout.addWidget(loadBtn)
        saveBtn = QPushButton("Save Visualization")
        saveBtn.clicked.connect(self.saveVisualization)
        topLayout.addWidget(saveBtn)
        exportDataBtn = QPushButton("Export Data (CSV)")
        exportDataBtn.clicked.connect(self.exportData)
        topLayout.addWidget(exportDataBtn)
        exportVideoBtn = QPushButton("Export Video")
        exportVideoBtn.clicked.connect(self.exportVideo)
        topLayout.addWidget(exportVideoBtn)
        popOutBtn = QPushButton("Pop Out Graphs")
        popOutBtn.clicked.connect(self.popOutGraphs)
        topLayout.addWidget(popOutBtn)
        visLayout.addLayout(topLayout)

        basicForm = QFormLayout()
        self.visTypeCombo = QComboBox()
        # New visualization modes including advanced features.
        self.visTypeCombo.addItems([
            "STFT (2D)",
            "STFT (Waterfall)",
            "STFT (Waterfall 3D)",
            "Mel (2D)",
            "CQT (2D)",
            "Waveform",
            "Chromagram",
            "MFCC",
            "Spectral Contrast",
            "Spectral Flux",
            "Tonnetz",
            "Audio Fingerprint",
            "Hybrid Visualization"
        ])
        basicForm.addRow("Visualization Mode:", self.visTypeCombo)
        self.nfftEdit = QLineEdit("1024")
        basicForm.addRow("n_fft:", self.nfftEdit)
        self.hopEdit = QLineEdit("256")
        basicForm.addRow("hop_length:", self.hopEdit)
        self.colorModeCombo = QComboBox()
        # Predefined color schemes for visual clarity.
        self.colorModeCombo.addItems([
            "Default (Viridis)",
            "Blue-Green",
            "Purple-Orange",
            "Rainbow",
            "Greyscale",
            "Negative Highlight"
        ])
        basicForm.addRow("Color Mode:", self.colorModeCombo)
        visLayout.addLayout(basicForm)

        btnLayout = QHBoxLayout()
        visBtn = QPushButton("Visualize Audio")
        visBtn.clicked.connect(self.visualizeAudio)
        btnLayout.addWidget(visBtn)
        statsBtn = QPushButton("Show Statistics")
        statsBtn.clicked.connect(self.showStatistics)
        btnLayout.addWidget(statsBtn)
        visLayout.addLayout(btnLayout)

        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        visLayout.addWidget(self.toolbar)
        visLayout.addWidget(self.canvas)

        self.tabs.addTab(visWidget, "Visualization")

        # Advanced Parameters Tab
        advWidget = QWidget()
        advLayout = QVBoxLayout()
        advWidget.setLayout(advLayout)
        advForm = QFormLayout()
        self.nMelsEdit = QLineEdit("128")
        advForm.addRow("n_mels (Mel):", self.nMelsEdit)
        self.nMfccEdit = QLineEdit("13")
        advForm.addRow("n_mfcc (MFCC):", self.nMfccEdit)
        self.nBandsEdit = QLineEdit("7")
        advForm.addRow("n_bands (Spectral Contrast):", self.nBandsEdit)
        self.logFreqCheck = QCheckBox("Use Log Frequency Axis")
        self.logFreqCheck.setChecked(True)
        advForm.addRow(self.logFreqCheck)
        self.overlayOnsetsCheck = QCheckBox("Overlay Onsets")
        advForm.addRow(self.overlayOnsetsCheck)
        self.overlayBeatsCheck = QCheckBox("Overlay Beats")
        advForm.addRow(self.overlayBeatsCheck)
        self.pitchTrackCheck = QCheckBox("Show Pitch Tracking")
        advForm.addRow(self.pitchTrackCheck)
        self.hpssCheck = QCheckBox("Harmonic/Percussive Separation")
        advForm.addRow(self.hpssCheck)
        self.zcrCheck = QCheckBox("Show Zero-Crossing Rate")
        advForm.addRow(self.zcrCheck)
        self.rmsCheck = QCheckBox("Show RMS Energy")
        advForm.addRow(self.rmsCheck)
        # Toggle to mark peak frequencies during visualization.
        self.flagModeCheck = QCheckBox("Flag Mode")
        advForm.addRow(self.flagModeCheck)
        self.entropyTypeCombo = QComboBox()
        self.entropyTypeCombo.addItems(["Shannon", "Renyi"])
        advForm.addRow("Entropy Type:", self.entropyTypeCombo)
        self.renyiOrderEdit = QLineEdit("2")
        advForm.addRow("Renyi Order:", self.renyiOrderEdit)
        # Dynamic control for adjusting spectral flux sensitivity.
        fluxSliderLabel = QLabel("Spectral Flux Sensitivity:")
        self.fluxSlider = QSlider(Qt.Horizontal)
        self.fluxSlider.setMinimum(1)
        self.fluxSlider.setMaximum(100)
        self.fluxSlider.setValue(50)
        advForm.addRow(fluxSliderLabel, self.fluxSlider)
        advLayout.addLayout(advForm)
        infoLabel = QLabel("Adjust advanced parameters and extra feature options.\nThese settings affect visualization and entropy calculation.")
        advLayout.addWidget(infoLabel)

        self.tabs.addTab(advWidget, "Advanced Parameters")

    def loadFile(self):
        # Opens a file dialog and loads the selected audio file into memory.
        debug_print("Opening file dialog to select audio file uwu~")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)"
        )
        if file_path:
            debug_print(f"File selected: {file_path}")
            self.audio_file = file_path
            self.fileLabel.setText(os.path.basename(file_path))
            try:
                debug_print("Loading audio using librosa.load()")
                self.y, self.sr = librosa.load(self.audio_file, sr=None)
                debug_print(f"Audio loaded: {len(self.y)} samples at {self.sr} Hz")
            except Exception as e:
                debug_print(f"Error loading audio: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load audio:\n{e}")
                self.audio_file = None
        else:
            debug_print("No file was selected uwu~")

    def getColormap(self, choice):
        # Returns a colormap object or string based on the selected scheme.
        debug_print(f"Getting colormap for: {choice}")
        colormap_mapping = {
            "Default (Viridis)": 'viridis',
            "Blue-Green": mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            "Purple-Orange": 'magma',
            "Rainbow": 'rainbow',
            "Greyscale": mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)], N=256),
            "Negative Highlight": 'seismic'
        }
        cmap = colormap_mapping.get(choice, 'viridis')
        debug_print(f"Using colormap: {cmap}")
        return cmap
        #j#a#n#e#r#a#i#n
    def _plot_main_visualization(self, vis_choice, cmap, n_fft, hop_length, log_freq, y_plot):
        # Core function to compute various visualizations based on mode.
        if vis_choice.startswith("STFT (2D)"):
            debug_print("Computing STFT (2D) uwu")
            D = librosa.stft(y_plot, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            self.current_spec_data = S_db
            times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=self.sr, hop_length=hop_length)
            return S_db, "STFT Spectrogram", times, librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        elif vis_choice.startswith("STFT (Waterfall)"):
            debug_print("Computing STFT for Waterfall mode")
            D = librosa.stft(y_plot, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            self.current_spec_data = S_db
            times = librosa.frames_to_time(np.arange(D.shape[1]), sr=self.sr, hop_length=hop_length)
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            return S_db, "STFT Spectrogram (Waterfall)", times, freqs
        elif vis_choice.startswith("STFT (Waterfall 3D)"):
            debug_print("Computing 3D Waterfall STFT")
            D = librosa.stft(y_plot, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            self.current_spec_data = S_db
            times = librosa.frames_to_time(np.arange(D.shape[1]), sr=self.sr, hop_length=hop_length)
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            return S_db, "STFT Spectrogram (Waterfall 3D)", times, freqs
        elif vis_choice.startswith("Mel (2D)"):
            debug_print("Computing Mel spectrogram")
            try:
                n_mels = int(self.nMelsEdit.text())
            except ValueError:
                n_mels = 128
            D = librosa.feature.melspectrogram(y=y_plot, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            self.current_spec_data = S_db
            times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=self.sr, hop_length=hop_length)
            return S_db, "Mel Spectrogram", times, None
        elif vis_choice.startswith("CQT (2D)"):
            debug_print("Computing Constant-Q Transform (CQT)")
            D = np.abs(librosa.cqt(y=y_plot, sr=self.sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1')))
            S_db = librosa.amplitude_to_db(D, ref=np.max)
            self.current_spec_data = S_db
            times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=self.sr, hop_length=hop_length)
            return S_db, "CQT Spectrogram", times, None
        elif vis_choice.startswith("Waveform"):
            debug_print("Preparing waveform display uwu")
            self.current_spec_data = None
            return None, "Waveform", None, None
        elif vis_choice.startswith("Chromagram"):
            debug_print("Computing Chromagram")
            D = librosa.feature.chroma_stft(y=y_plot, sr=self.sr, n_fft=n_fft, hop_length=hop_length)
            self.current_spec_data = D
            times = librosa.frames_to_time(np.arange(D.shape[1]), sr=self.sr, hop_length=hop_length)
            return D, "Chromagram", times, None
        elif vis_choice.startswith("MFCC"):
            debug_print("Computing MFCC features")
            try:
                n_mfcc = int(self.nMfccEdit.text())
            except ValueError:
                n_mfcc = 13
            D = librosa.feature.mfcc(y=y_plot, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
            self.current_spec_data = D
            times = librosa.frames_to_time(np.arange(D.shape[1]), sr=self.sr, hop_length=hop_length)
            return D, "MFCC", times, None
        elif vis_choice.startswith("Spectral Contrast"):
            debug_print("Computing Spectral Contrast features")
            try:
                n_bands = int(self.nBandsEdit.text())
            except ValueError:
                n_bands = 7
            D = librosa.feature.spectral_contrast(y=y_plot, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
            self.current_spec_data = D
            times = librosa.frames_to_time(np.arange(D.shape[1]), sr=self.sr, hop_length=hop_length)
            return D, "Spectral Contrast", times, None
        elif vis_choice.startswith("Spectral Flux"):
            debug_print("Computing Spectral Flux")
            # Compute magnitude spectrogram
            D = np.abs(librosa.stft(y_plot, n_fft=n_fft, hop_length=hop_length))
            # Calculate positive differences between frames
            flux = np.diff(D, axis=1)
            flux = np.maximum(0, flux)
            # Sum flux across frequency bins
            flux = np.sum(flux, axis=0)
            # Apply sensitivity adjustment using the slider value
            sensitivity = self.fluxSlider.value() / 50.0
            flux *= sensitivity
            flux_img = np.expand_dims(flux, axis=0)  # Make it 2D for display
            self.current_spec_data = flux_img
            times = librosa.frames_to_time(np.arange(flux_img.shape[1]), sr=self.sr, hop_length=hop_length)
            return flux_img, "Spectral Flux", times, [0, 1]
        elif vis_choice.startswith("Tonnetz"):
            debug_print("Computing Tonnetz features (serious analysis)")
            y_harmonic = librosa.effects.hpss(y_plot)[0]
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=self.sr)
            self.current_spec_data = tonnetz
            times = librosa.frames_to_time(np.arange(tonnetz.shape[1]), sr=self.sr, hop_length=hop_length)
            freqs = np.linspace(0, tonnetz.shape[0], tonnetz.shape[0])
            return tonnetz, "Tonnetz Features", times, freqs
        elif vis_choice.startswith("Audio Fingerprint"):
            debug_print("Computing Audio Fingerprint (advanced)")
            try:
                n_mfcc = int(self.nMfccEdit.text())
            except ValueError:
                n_mfcc = 13
            mfcc = librosa.feature.mfcc(y=y_plot, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
            self.current_spec_data = mfcc
            fingerprint = hash(np.round(mfcc, decimals=1).tobytes())
            title = f"Audio Fingerprint: {fingerprint}"
            times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=self.sr, hop_length=hop_length)
            return mfcc, title, times, None
        elif vis_choice.startswith("Hybrid Visualization"):
            # Hybrid mode is managed separately in visualizeAudio.
            raise ValueError("Hybrid Visualization is handled separately.")
        else:
            raise ValueError("Unsupported visualization mode selected.")

    def _set_axis_limits(self, ax, times=None, freqs=None):
        # Configures axis limits based on provided time and frequency arrays.
        if times is not None:
            ax.set_xlim(times[0], times[-1])
        if freqs is not None:
            ax.set_ylim(freqs[0], freqs[-1])
        ax.relim()
        ax.autoscale_view()

    def visualizeAudio(self):
        # Main function to compute and render the visualization.
        debug_print("Visualize Audio button clicked")
        if self.audio_file is None or self.y is None:
            debug_print("No audio file loaded; aborting visualization uwu~")
            QMessageBox.warning(self, "Warning", "Please load an audio file first.")
            return

        vis_choice = self.visTypeCombo.currentText()
        color_mode = self.colorModeCombo.currentText()
        cmap = self.getColormap(color_mode)
        log_freq = self.logFreqCheck.isChecked()
        try:
            n_fft = int(self.nfftEdit.text())
            hop_length = int(self.hopEdit.text())
        except ValueError:
            debug_print("Invalid FFT parameters provided")
            QMessageBox.warning(self, "Warning", "Enter valid integers for FFT parameters.")
            return

        y_plot = librosa.effects.hpss(self.y)[0] if self.hpssCheck.isChecked() else self.y
        #https://github.com/r4debox
        # Handle Hybrid Visualization separately due to its multi-panel layout.
        if vis_choice.startswith("Hybrid Visualization"):
            try:
                stft_data, stft_title, stft_times, stft_freqs = self._plot_main_visualization("STFT (2D)", cmap, n_fft, hop_length, log_freq, y_plot)
                mfcc_data, mfcc_title, mfcc_times, _ = self._plot_main_visualization("MFCC", cmap, n_fft, hop_length, log_freq, y_plot)
                tonnetz_data, tonnetz_title, tonnetz_times, _ = self._plot_main_visualization("Tonnetz", cmap, n_fft, hop_length, log_freq, y_plot)
            except Exception as e:
                debug_print(f"Error computing hybrid features: {e}")
                QMessageBox.critical(self, "Error", f"Error computing hybrid visualization:\n{e}")
                return
            self.figure.clf()
            ax1 = self.figure.add_subplot(311)
            extent1 = [stft_times[0], stft_times[-1], stft_freqs[0], stft_freqs[-1]]
            img1 = ax1.imshow(stft_data, origin='lower', aspect='auto', extent=extent1, cmap=cmap)
            self.figure.colorbar(img1, ax=ax1, format='%+2.0f dB')
            ax1.set_title(stft_title)
            ax2 = self.figure.add_subplot(312)
            extent2 = [mfcc_times[0], mfcc_times[-1], 0, mfcc_data.shape[0]]
            img2 = ax2.imshow(mfcc_data, origin='lower', aspect='auto', extent=extent2, cmap=cmap)
            self.figure.colorbar(img2, ax=ax2)
            ax2.set_title(mfcc_title)
            ax3 = self.figure.add_subplot(313)
            extent3 = [tonnetz_times[0], tonnetz_times[-1], 0, tonnetz_data.shape[0]]
            img3 = ax3.imshow(tonnetz_data, origin='lower', aspect='auto', extent=extent3, cmap=cmap)
            self.figure.colorbar(img3, ax=ax3)
            ax3.set_title(tonnetz_title)
            self.figure.tight_layout()
            self.canvas.draw()
            return

        try:
            data, main_title, times, freqs = self._plot_main_visualization(vis_choice, cmap, n_fft, hop_length, log_freq, y_plot)
        except Exception as e:
            debug_print(f"Error computing main visualization: {e}")
            QMessageBox.critical(self, "Error", f"Error computing visualization:\n{e}")
            return

        # Compute spectral entropy over time using frame-based FFT analysis.
        entropy_type = self.entropyTypeCombo.currentText()
        if entropy_type == "Renyi":
            try:
                alpha = float(self.renyiOrderEdit.text())
                if alpha == 1:
                    raise ValueError("Renyi order cannot be 1.")
            except ValueError:
                alpha = 2.0
        else:
            alpha = None
        entropy_list = []
        for i in range(0, len(self.y) - n_fft, hop_length):
            frame = self.y[i: i+n_fft]
            frame_fft = np.fft.fft(frame)
            power = np.abs(frame_fft)**2
            total = np.sum(power)
            if total == 0:
                entropy_list.append(0)
                continue
            probs = power / total
            if entropy_type == "Renyi":
                ent = 1/(1-alpha) * np.log(np.sum(probs**alpha))
            else:
                ent = scipy.stats.entropy(probs)
            entropy_list.append(ent)
        entropy_arr = np.array(entropy_list)
        times_entropy = librosa.times_like(entropy_arr, sr=self.sr, hop_length=hop_length)
        self.entropy_data = entropy_arr
        self.entropy_times = times_entropy

        extras = []
        if self.zcrCheck.isChecked():
            debug_print("Computing Zero-Crossing Rate")
            zcr = librosa.feature.zero_crossing_rate(y_plot, hop_length=hop_length)[0]
            times_zcr = librosa.times_like(zcr, sr=self.sr, hop_length=hop_length)
            extras.append(("Zero-Crossing Rate", times_zcr, zcr))
        if self.rmsCheck.isChecked():
            debug_print("Computing RMS Energy")
            rms = librosa.feature.rms(y=y_plot, hop_length=hop_length)[0]
            times_rms = librosa.times_like(rms, sr=self.sr, hop_length=hop_length)
            extras.append(("RMS Energy", times_rms, rms))
        self.extra_plots_data = extras

        self.figure.clf()
        # Use 3D branch for STFT (Waterfall 3D) mode.
        if vis_choice == "STFT (Waterfall 3D)":
            total_rows = 1 + 1 + len(extras)
            gs = self.figure.add_gridspec(total_rows, 1)
            ax_main = self.figure.add_subplot(gs[0, 0], projection='3d')
            if freqs is None:
                freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            X, Y = np.meshgrid(times, freqs)
            norm = TwoSlopeNorm(vmin=np.min(self.current_spec_data), vcenter=-40, vmax=0) if color_mode == "Negative Highlight" else None
            surf = ax_main.plot_surface(X, Y, self.current_spec_data, cmap=cmap, norm=norm)
            self.figure.colorbar(surf, ax=ax_main, shrink=0.5, aspect=10)
            ax_main.set_title(main_title)
            row_idx = 1
            if self.flagModeCheck.isChecked():
                for i in range(self.current_spec_data.shape[1]):
                    col = self.current_spec_data[:, i]
                    peak_idx = np.argmax(col)
                    time_val = times[i]
                    freq_val = freqs[peak_idx]
                    z_val = self.current_spec_data[peak_idx, i]
                    ax_main.scatter(time_val, freq_val, z_val, color='red', s=10)
            ax_entropy = self.figure.add_subplot(gs[row_idx, 0])
            ax_entropy.plot(times_entropy, entropy_arr, color='tab:orange')
            ax_entropy.set_title("Spectral Entropy Over Time")
            ax_entropy.set_xlabel("Time (s)")
            ax_entropy.set_ylabel("Entropy")
            ax_entropy.grid(True)
            row_idx += 1
            for title, t_arr, data_arr in extras:
                ax_extra = self.figure.add_subplot(gs[row_idx, 0])
                ax_extra.plot(t_arr, data_arr, color='green')
                ax_extra.set_title(title)
                ax_extra.set_xlabel("Time (s)")
                row_idx += 1
        else:
            total_rows = 2 + len(extras)
            axes = self.figure.subplots(total_rows, 1, sharex=True)
            if total_rows == 1:
                axes = [axes]
            ax_main = axes[0]
            if vis_choice.startswith("Waveform"):
                times_wave = np.linspace(0, len(y_plot)/self.sr, num=len(y_plot))
                ax_main.plot(times_wave, y_plot, color='steelblue')
            else:
                if freqs is None:
                    freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
                extent = [times[0], times[-1], freqs[0], freqs[-1]]
                norm = TwoSlopeNorm(vmin=np.min(self.current_spec_data), vcenter=-40, vmax=0) if color_mode == "Negative Highlight" else None
                img = ax_main.imshow(self.current_spec_data, origin='lower', aspect='auto', extent=extent, cmap=cmap, norm=norm)
                self.figure.colorbar(img, ax=ax_main, format='%+2.0f dB')
                self._set_axis_limits(ax_main, times, freqs)
            ax_main.set_title(main_title)
            if self.flagModeCheck.isChecked() and vis_choice.startswith("STFT"):
                for i in range(self.current_spec_data.shape[1]):
                    col = self.current_spec_data[:, i]
                    peak_idx = np.argmax(col)
                    time_val = times[i]
                    freq_val = freqs[peak_idx]
                    ax_main.plot(time_val, freq_val, marker='o', color='red', markersize=4)
            ax_entropy = axes[1]
            ax_entropy.plot(times_entropy, entropy_arr, color='tab:orange')
            ax_entropy.set_title("Spectral Entropy Over Time")
            ax_entropy.set_xlabel("Time (s)")
            ax_entropy.set_ylabel("Entropy")
            ax_entropy.grid(True)
            for idx, (title, t_arr, data_arr) in enumerate(extras, start=2):
                ax_extra = axes[idx]
                ax_extra.plot(t_arr, data_arr, color='green')
                ax_extra.set_title(title)
                ax_extra.set_xlabel("Time (s)")
        self.figure.tight_layout()
        self.canvas.draw()

    def popOutGraphs(self):
        # Creates separate windows for each graph.
        debug_print("Pop Out Graphs button clicked uwu~")
        if self.current_spec_data is not None:
            fig_main = plt.figure()
            if self.visTypeCombo.currentText() == "STFT (Waterfall 3D)":
                ax_main = fig_main.add_subplot(111, projection='3d')
                try:
                    n_fft = int(self.nfftEdit.text())
                except ValueError:
                    n_fft = 1024
                freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
                times = librosa.frames_to_time(np.arange(self.current_spec_data.shape[1]), sr=self.sr, hop_length=int(self.hopEdit.text()))
                X, Y = np.meshgrid(times, freqs)
                ax_main.plot_surface(X, Y, self.current_spec_data, cmap=self.getColormap(self.colorModeCombo.currentText()))
            else:
                ax_main = fig_main.add_subplot(111)
                img = librosa.display.specshow(self.current_spec_data, sr=self.sr, x_axis='time', cmap=self.getColormap(self.colorModeCombo.currentText()), ax=ax_main)
                ax_main.set_aspect('auto')
                fig_main.colorbar(img, ax=ax_main, format='%+2.0f dB')
            ax_main.set_title("Main Visualization")
            fig_main.show()
        elif self.y is not None:
            fig_wave, ax_wave = plt.subplots()
            times_wave = np.linspace(0, len(self.y)/self.sr, num=len(self.y))
            ax_wave.plot(times_wave, self.y, color='steelblue')
            ax_wave.set_title("Waveform")
            fig_wave.show()
        if self.entropy_data is not None and self.entropy_times is not None:
            fig_entropy, ax_entropy = plt.subplots()
            ax_entropy.plot(self.entropy_times, self.entropy_data, color='tab:orange')
            ax_entropy.set_title("Spectral Entropy Over Time")
            ax_entropy.set_xlabel("Time (s)")
            ax_entropy.set_ylabel("Entropy")
            ax_entropy.grid(True)
            fig_entropy.show()
        for title, t_arr, data_arr in self.extra_plots_data:
            fig_extra, ax_extra = plt.subplots()
            ax_extra.plot(t_arr, data_arr, color='green')
            ax_extra.set_title(title)
            ax_extra.set_xlabel("Time (s)")
            fig_extra.show()
        debug_print("Popped out all graphs into separate windows.")

    def showStatistics(self):
        # Displays basic statistics of the computed data.
        if self.current_spec_data is None:
            QMessageBox.information(self, "Statistics", "No computed data available for statistics.")
            return
        data = self.current_spec_data
        stats_text = (
            f"Min: {np.min(data):.2f}\n"
            f"Max: {np.max(data):.2f}\n"
            f"Mean: {np.mean(data):.2f}\n"
            f"Std Dev: {np.std(data):.2f}"
        )
        debug_print("Showing statistics for computed data")
        QMessageBox.information(self, "Data Statistics", stats_text)

    def saveVisualization(self):
        # Saves the current visualization to an image file.
        debug_print("Save Visualization button clicked uwu~")
        fname, _ = QFileDialog.getSaveFileName(self, "Save Visualization", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if fname:
            try:
                self.figure.savefig(fname)
                debug_print(f"Visualization saved to {fname}")
                QMessageBox.information(self, "Save Visualization", f"Visualization saved to:\n{fname}")
            except Exception as e:
                debug_print(f"Error saving visualization: {e}")
                QMessageBox.critical(self, "Error", f"Error saving visualization:\n{e}")

    def exportData(self):
        # Exports computed data to a CSV file.
        if self.current_spec_data is None:
            QMessageBox.information(self, "Export Data", "No computed data to export.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export Data as CSV", "", "CSV Files (*.csv)")
        if fname:
            try:
                with open(fname, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    for row in self.current_spec_data:
                        writer.writerow(row)
                debug_print(f"Data exported to {fname}")
                QMessageBox.information(self, "Export Data", f"Data exported to:\n{fname}")
            except Exception as e:
                debug_print(f"Error exporting data: {e}")
                QMessageBox.critical(self, "Error", f"Error exporting data:\n{e}")

    def exportVideo(self):
        # Generates and saves a video animation of the STFT visualization.
        #janerain
        debug_print("Export Video button clicked uwu~")
        if self.y is None:
            QMessageBox.information(self, "Export Video", "No audio loaded to export video from.")
            return
        try:
            n_fft = int(self.nfftEdit.text())
            hop_length = int(self.hopEdit.text())
        except ValueError:
            QMessageBox.warning(self, "Export Video", "Invalid FFT parameters for video export.")
            return
        D_full = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length)
        S_db_full = librosa.amplitude_to_db(np.abs(D_full), ref=np.max)
        times = librosa.frames_to_time(np.arange(D_full.shape[1]), sr=self.sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        fig_anim, ax_anim = plt.subplots(figsize=(10, 4))
        img_anim = ax_anim.imshow(S_db_full[:, :1], aspect='auto', origin='lower',
                                  extent=[times[0], times[0]+0.1, freqs[0], freqs[-1]],
                                  cmap="viridis")
        ax_anim.set_xlabel("Time (s)")
        ax_anim.set_ylabel("Frequency (Hz)")
        ax_anim.set_title("STFT Spectrogram Animation")
        def update_frame(i):
            current_time = times[i]
            img_anim.set_data(S_db_full[:, :i+1])
            img_anim.set_extent([times[0], current_time, freqs[0], freqs[-1]])
            return [img_anim]
        ani = animation.FuncAnimation(fig_anim, update_frame, frames=len(times), interval=50, blit=True)
        out_fname, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Video (*.mp4)")
        if out_fname:
            try:
                ani.save(out_fname, writer='ffmpeg')
                debug_print(f"Video exported to {out_fname}")
                QMessageBox.information(self, "Export Video", f"Video exported to:\n{out_fname}")
            except Exception as e:
                debug_print(f"Error exporting video: {e}")
                QMessageBox.critical(self, "Error", f"Error exporting video:\n{e}")
        plt.close(fig_anim)

def main():
    # Entry point for the application. Sets up QApplication and shows the main window.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        window = AudioVisualizer()
        window.resize(1000, 800)
        window.show()
        debug_print("Application window shown. Entering main loop.")
        sys.exit(app.exec_())
    else:
        window = AudioVisualizer()
        window.resize(1000, 800)
        window.show()
        debug_print("Application window shown (existing event loop).")

if __name__ == "__main__":
    # Process command-line arguments for debug file logging.
    for arg in sys.argv[1:]:
        if arg.startswith("--debugfile="):
            DEBUG_FILE = arg.split("=", 1)[1]
    main()
