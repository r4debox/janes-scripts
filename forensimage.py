#!/usr/bin/env python3
# This script is released into the public domain.
# Anyone is free to use, modify, and distribute this software.
# For more information, please refer to <https://unlicense.org/>.

import sys
import os
import re
import io
import hashlib
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QLineEdit, QFileDialog, QMessageBox, QFormLayout, QTabWidget,
    QSlider, QGroupBox, QRadioButton, QListWidget
)
from PyQt5.QtCore import Qt

# Additional imports for advanced functions
from scipy.fftpack import dct
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter, gaussian_filter

try:
    from skimage.feature import local_binary_pattern
except ImportError:
    local_binary_pattern = None

def debug_print(message):
    print(f"[DEBUG] {message}")

def my_exception_hook(exctype, value, traceback_obj):
    import traceback
    err_msg = "Uncaught exception: " + str(value)
    print(err_msg)
    traceback.print_exception(exctype, value, traceback_obj)
    input("Press Enter to exit...")
    sys.exit(1)

sys.excepthook = my_exception_hook

# ------------------ Main Class ------------------
class ImageVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Visualizer and Analyzer")
        self.image_file = None      # Path to loaded image file
        self.image = None           # PIL Image object
        self.current_data = None    # Current visualization data (numpy array)
        debug_print("Initializing ImageVisualizer")
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Visualization Tab 0_0---
        visWidget = QWidget()
        visLayout = QVBoxLayout()
        visWidget.setLayout(visLayout)

        # Top control panel (file label, load and save buttons)
        topLayout = QHBoxLayout()
        self.fileLabel = QLabel("No file selected")
        loadBtn = QPushButton("Load Image")
        loadBtn.clicked.connect(self.loadFile)
        topLayout.addWidget(self.fileLabel)
        topLayout.addWidget(loadBtn)
        saveBtn = QPushButton("Save Visualization")
        saveBtn.clicked.connect(self.saveVisualization)
        topLayout.addWidget(saveBtn)
        visLayout.addLayout(topLayout)

        # Form for visualization mode and basic parameters
        formLayout = QFormLayout()
        self.visTypeCombo = QComboBox()
        # List of all individual visualization modes (scrolls)
        self.visTypeCombo.addItems([
            "Original Image",
            "Grayscale",
            "Histogram",
            "Edge Detection",
            "Fourier Transform",
            "Artifact Detection",
            "Error Level Analysis",
            "ELA",
            "Noise Analysis",
            "Level Sweep",
            "Luminance",
            "Gradient",
            "Principal Component Analysis",
            "Meta Data",
            "Geo Tags",
            "Thumbnail Analysis",
            "JPEG Analysis",
            "String Extraction",
            "Clone Detection",
            "Digest",
            "Hidden Pixels",
            "ICC+",
            "JPEG %",
            "Metadata",
            "Strings",
            "Color Channel Analysis",
            "Saturation Map",
            "Image Entropy",
            "Histogram Equalization",
            "Dominant Color Extraction",
            "Contrast Map",
            "Edge Orientation Map",
            "DCT Analysis"
        ])
        formLayout.addRow("Visualization Mode:", self.visTypeCombo)
        self.histBinsEdit = QLineEdit("256")
        formLayout.addRow("Histogram Bins:", self.histBinsEdit)
        self.stringMinLengthEdit = QLineEdit("4")
        formLayout.addRow("Min String Length:", self.stringMinLengthEdit)
        visLayout.addLayout(formLayout)

        # Button panel 
        btnLayout = QHBoxLayout()
        visBtn = QPushButton("Visualize Image")
        visBtn.clicked.connect(self.visualizeImage)
        btnLayout.addWidget(visBtn)
        popOutBtn = QPushButton("Pop Out Graphs")
        popOutBtn.clicked.connect(self.popOutGraphs)
        btnLayout.addWidget(popOutBtn)
        visLayout.addLayout(btnLayout)

        # Matplotlib Figure and Canvas >////<
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        visLayout.addWidget(self.toolbar)
        visLayout.addWidget(self.canvas)
        self.tabs.addTab(visWidget, "Visualization")

        # --- Advanced Parameters Tab (Sequential Analysis) ---
        self.tabs.addTab(self.initAdvancedParametersTab(), "Sequential analysis")

        # --- Forensics Parameters Tab (now with sliders!!!!) ---
        self.tabs.addTab(self.initForensicsParametersTab(), "Forensics Parameters")
        
        # --- Help Tab ---
        self.tabs.addTab(self.initHelpTab(), "Help")

    # ------------------ Advanced Parameters Tab (Sequential Analysis) ------------------
    def initAdvancedParametersTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        groupBox = QGroupBox("Sequential Analysis Options")
        groupLayout = QVBoxLayout()

        # Radio buttons: Predefined vs. Custom pipeline
        self.radioPredefined = QRadioButton("Predefined Pipeline")
        self.radioCustom = QRadioButton("Custom Pipeline")
        self.radioPredefined.setChecked(True)
        radioLayout = QHBoxLayout()
        radioLayout.addWidget(self.radioPredefined)
        radioLayout.addWidget(self.radioCustom)
        groupLayout.addLayout(radioLayout)

        # Predefined pipeline combo box (only chainable modes)
        self.predefinedCombo = QComboBox()
        self.predefinedCombo.addItems([
            "Original Image -> Grayscale -> Edge Detection",
            "Original Image -> Fourier Transform -> Artifact Detection",
            "Original Image -> Grayscale -> Histogram Equalization -> Contrast Map",
            "Original Image -> Grayscale -> Histogram Equalization -> Contrast Map -> Edge Orientation Map",
            "Original Image -> Grayscale -> Edge Detection -> Contrast Map -> Edge Orientation Map",  # Edge Focus
            "Original Image -> Grayscale -> Histogram Equalization",  # Texture Analysis
            "Original Image -> Fourier Transform -> DCT Analysis",  # Frequency Analysis
            "Original Image -> Grayscale -> Histogram Equalization -> Contrast Map -> Edge Detection",  # Combined Enhancement
            "Histogram Equalization -> Edge Detection -> Luminance -> Artifact Detection -> Contrast Map -> Edge Orientation Map" # Kitchen Sink
        ])
        groupLayout.addWidget(QLabel("Predefined Pipeline:"))
        groupLayout.addWidget(self.predefinedCombo)

        # Custom pipeline builder (available steps are only chainable modes)
        self.availableSequentialModes = ["Original Image", "Grayscale", "Histogram Equalization",
                                         "Edge Detection", "Fourier Transform", "Artifact Detection",
                                         "Luminance", "Contrast Map", "Edge Orientation Map",
                                         "LBP Analysis", "Saturation Map", "DCT Analysis", "Gradient"]
        self.customModeCombo = QComboBox()
        self.customModeCombo.addItems(self.availableSequentialModes)
        self.btnAddStep = QPushButton("Add Step")
        self.btnRemoveStep = QPushButton("Remove Selected Step")
        self.customPipelineList = QListWidget()
        customBtnLayout = QHBoxLayout()
        customBtnLayout.addWidget(self.btnAddStep)
        customBtnLayout.addWidget(self.btnRemoveStep)
        groupLayout.addWidget(QLabel("Available Steps (for Custom Pipeline):"))
        groupLayout.addWidget(self.customModeCombo)
        groupLayout.addLayout(customBtnLayout)
        groupLayout.addWidget(QLabel("Custom Pipeline Steps:"))
        groupLayout.addWidget(self.customPipelineList)

        # Connect custom pipeline buttons
        self.btnAddStep.clicked.connect(self.addCustomStep)
        self.btnRemoveStep.clicked.connect(self.removeCustomStep)

        # Run Sequential Analysis button
        self.btnRunSequence = QPushButton("Run Sequential Analysis")
        self.btnRunSequence.clicked.connect(self.runSequentialAnalysis)
        groupLayout.addWidget(self.btnRunSequence)

        groupBox.setLayout(groupLayout)
        layout.addWidget(groupBox)

        # Update visibility based on radio button selection
        self.radioPredefined.toggled.connect(self.updateSequentialOptionsVisibility)
        self.updateSequentialOptionsVisibility()

        return widget

    def updateSequentialOptionsVisibility(self):
        if self.radioPredefined.isChecked():
            self.predefinedCombo.setEnabled(True)
            self.customModeCombo.setEnabled(False)
            self.btnAddStep.setEnabled(False)
            self.btnRemoveStep.setEnabled(False)
            self.customPipelineList.setEnabled(False)
        else:
            self.predefinedCombo.setEnabled(False)
            self.customModeCombo.setEnabled(True)
            self.btnAddStep.setEnabled(True)
            self.btnRemoveStep.setEnabled(True)
            self.customPipelineList.setEnabled(True)

    def addCustomStep(self):
        step = self.customModeCombo.currentText()
        self.customPipelineList.addItem(step)

    def removeCustomStep(self):
        for item in self.customPipelineList.selectedItems():
            self.customPipelineList.takeItem(self.customPipelineList.row(item))

    # ------------------ Forensics Parameters Tab ------------------
    def initForensicsParametersTab(self):
        widget = QWidget()
        layout = QFormLayout()
        widget.setLayout(layout)
        self.sliderArtifactThreshold = QSlider(Qt.Horizontal)
        self.sliderArtifactThreshold.setMinimum(0)
        self.sliderArtifactThreshold.setMaximum(255)
        self.sliderArtifactThreshold.setValue(10)
        layout.addRow("Artifact Threshold:", self.sliderArtifactThreshold)
        self.sliderCloneBlockSize = QSlider(Qt.Horizontal)
        self.sliderCloneBlockSize.setMinimum(8)
        self.sliderCloneBlockSize.setMaximum(128)
        self.sliderCloneBlockSize.setValue(32)
        layout.addRow("Clone Block Size:", self.sliderCloneBlockSize)
        self.sliderBlurRadius = QSlider(Qt.Horizontal)
        self.sliderBlurRadius.setMinimum(1)
        self.sliderBlurRadius.setMaximum(10)
        self.sliderBlurRadius.setValue(1)
        layout.addRow("Blur Radius:", self.sliderBlurRadius)
        self.sliderLBPRadius = QSlider(Qt.Horizontal)
        self.sliderLBPRadius.setMinimum(1)
        self.sliderLBPRadius.setMaximum(10)
        self.sliderLBPRadius.setValue(1)
        layout.addRow("LBP Radius:", self.sliderLBPRadius)
        self.sliderLBPNeighbors = QSlider(Qt.Horizontal)
        self.sliderLBPNeighbors.setMinimum(8)
        self.sliderLBPNeighbors.setMaximum(24)
        self.sliderLBPNeighbors.setValue(8)
        layout.addRow("LBP Neighbors:", self.sliderLBPNeighbors)
        self.sliderDCTBlockSize = QSlider(Qt.Horizontal)
        self.sliderDCTBlockSize.setMinimum(8)
        self.sliderDCTBlockSize.setMaximum(64)
        self.sliderDCTBlockSize.setValue(16)
        layout.addRow("DCT Block Size:", self.sliderDCTBlockSize)
        self.sliderEdgeDensityWindow = QSlider(Qt.Horizontal)
        self.sliderEdgeDensityWindow.setMinimum(3)
        self.sliderEdgeDensityWindow.setMaximum(31)
        self.sliderEdgeDensityWindow.setValue(7)
        layout.addRow("Edge Density Window:", self.sliderEdgeDensityWindow)
        return widget

    # ------------------ Load File ------------------
    def loadFile(self):
        debug_print("Opening file dialog to select image file")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            debug_print(f"File selected: {file_path}")
            self.image_file = file_path
            self.fileLabel.setText(os.path.basename(file_path))
            try:
                self.image = Image.open(self.image_file)
                debug_print(f"Image loaded: {self.image.size}, mode: {self.image.mode}, format: {self.image.format}")
            except Exception as e:
                debug_print(f"Error loading image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
                self.image_file = None
        else:
            debug_print("No file was selected")

    # ----------------- Help Tab -----------------
    def initHelpTab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
    
        help_text = (
        "Image Visualizer and Analyzer Help\n"
        "====================================\n\n"
        "This application allows you to load an image and process it using a variety of modes. \n"
        "Modes include basic processing (e.g., Grayscale, Histogram, Edge Detection) and advanced forensic analyses \n"
        "(e.g., Fourier Transform, Artifact Detection, Error Level Analysis, DCT Analysis, etc.).\n\n"
        "Individual Visualization Modes (Scroll for options):\n"
        "--------------------------------\n"
        "  • Original Image: Displays the loaded image as is.\n"
        "  • Grayscale: Converts the image to grayscale.\n"
        "  • Histogram: Plots the image's histogram (color channels for RGB or intensity for grayscale).\n"
        "  • Edge Detection: Highlights edges using a built-in edge detection filter.\n"
        "  • Fourier Transform: Shows the magnitude spectrum of the Fourier transform.\n"
        "  • Artifact Detection: Highlights compression or other artifacts in bright pink.\n"
        "  • Error Level Analysis (ELA): Compares the image with a recompressed JPEG version to reveal areas of loss.\n"
        "  • Noise Analysis: Displays the difference between the original and a blurred version to reveal noise.\n"
        "  • Level Sweep: Shows multiple brightness-adjusted versions of the image.\n"
        "  • Luminance: Computes and displays the luminance channel.\n"
        "  • Gradient: Computes the gradient magnitude (edge strength) of the image.\n"
        "  • Principal Component Analysis (PCA): Shows the first principal component of the color data.\n"
        "  • Meta Data / Metadata: Extracts and displays EXIF metadata if available.\n"
        "  • Geo Tags: Extracts GPS information from the image's EXIF data.\n"
        "  • Thumbnail Analysis: Checks for an embedded thumbnail in EXIF data.\n"
        "  • JPEG Analysis: Displays JPEG quantization tables if available.\n"
        "  • String Extraction / Strings: Extracts printable strings from the image file.\n"
        "  • Clone Detection: Identifies duplicate regions in the image.\n"
        "  • Digest: Computes a SHA-256 digest of the image file.\n"
        "  • Hidden Pixels: Extracts the least significant bits from the image pixels.\n"
        "  • ICC+: Extracts basic ICC profile information from the image.\n"
        "  • JPEG %: Estimates the compression ratio of the JPEG image.\n"
        "  • Color Channel Analysis: Separates and displays the individual R, G, and B channels.\n"
        "  • Saturation Map: Converts the image to HSV and shows the saturation channel.\n"
        "  • Image Entropy: Computes the entropy (information content) of the image.\n"
        "  • Histogram Equalization: Enhances contrast by equalizing the histogram.\n"
        "  • Dominant Color Extraction: Finds and displays the most frequent color in the image.\n"
        "  • Contrast Map: Computes a local contrast map based on standard deviation.\n"
        "  • Edge Orientation Map: Displays the orientation of edges in the image.\n"
        "  • DCT Analysis: Divides the image into blocks and computes DCT energy for frequency analysis.\n\n"
        "Sequential Analysis (Advanced Parameters Tab):\n"
        "-------------------------------------------------\n"
        "The Advanced Parameters tab lets you build a pipeline of chainable operations. \n"
        "Choose a predefined pipeline from the combo box or build your own custom pipeline using the available steps. \n"
        "Only modes that output an image (chainable modes) can be used here. Supported sequential modes are:\n"
        "   Original Image, Grayscale, Histogram Equalization, Edge Detection, Fourier Transform, \n"
        "   Artifact Detection, Luminance, Contrast Map, Edge Orientation Map, LBP Analysis, Saturation Map, \n"
        "   DCT Analysis, and Gradient.\n\n"
        "Forensics Parameters (Forensics Parameters Tab):\n"
        "------------------------------------------------\n"
        "Use the sliders to adjust thresholds and other parameters for various analysis modes (e.g., Artifact Detection, Clone Detection).\n\n"
        "General Usage Tips:\n"
        "-------------------\n"
        "  1. Load an image using the 'Load Image' button.\n"
        "  2. Select an individual mode from the Visualization tab and click 'Visualize Image' to see the result.\n"
        "  3. Use 'Pop Out Graphs' to open the current visualization in a new window.\n"
        "  4. In the Advanced Parameters tab, choose a sequential pipeline (predefined or custom) and click 'Run Sequential Analysis' \n"
        "     to process the image through multiple steps, with each step's output shown as a subplot.\n"
        "  5. Adjust parameters in the Forensics Parameters tab to fine-tune the analyses.\n\n"
        "This tool is intended for image forensics and analysis. All operations are performed using Python libraries. \n"
        "It is provided as-is in the public domain. Feel free to modify and extend its functionality as needed.\n"
        "Say hi to me on discord @janewind :)\n"
    )
    
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)  # Ensures long text wraps nicely ^.^
        layout.addWidget(help_label)
    
        return widget

    # ------------------ Sequential Analysis Helper Function ------------------
    def process_mode(self, mode, input_img):
        """
        Process the input PIL image according to the mode and return a new PIL image.
        Only chainable operations are allowed. Supported sequential modes are:
          Original Image, Grayscale, Histogram Equalization, Edge Detection, Fourier Transform,
          Artifact Detection, Luminance, Contrast Map, Edge Orientation Map, LBP Analysis,
          Saturation Map, DCT Analysis, and Gradient.
        """
        supported_modes = ["Original Image", "Grayscale", "Histogram Equalization", "Edge Detection",
                           "Fourier Transform", "Artifact Detection", "Luminance", "Contrast Map",
                           "Edge Orientation Map", "LBP Analysis", "Saturation Map", "DCT Analysis", "Gradient"]
        if mode not in supported_modes:
            raise Exception(f"Mode '{mode}' is not supported in sequential analysis. Supported modes are: {supported_modes}")
        debug_print(f"Processing mode: {mode}")
        try:
            if mode == "Original Image":
                return input_img
            elif mode == "Grayscale":
                return ImageOps.grayscale(input_img)
            elif mode == "Histogram Equalization":
                return ImageOps.equalize(input_img.convert("L"))
            elif mode == "Edge Detection":
                return input_img.filter(ImageFilter.FIND_EDGES)
            elif mode == "Fourier Transform":
                gray = ImageOps.grayscale(input_img)
                arr = np.array(gray)
                f_transform = np.fft.fft2(arr)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = 20 * np.log(np.abs(f_shift) + 1)
                magnitude = np.clip(magnitude, 0, 255)
                return Image.fromarray(np.uint8(magnitude))
            elif mode == "Artifact Detection":
                gray = ImageOps.grayscale(input_img)
                arr = np.array(gray).astype(np.float32)
                median_img = gray.filter(ImageFilter.MedianFilter(size=3))
                median_arr = np.array(median_img).astype(np.float32)
                diff = np.abs(arr - median_arr)
                threshold = self.sliderArtifactThreshold.value() if hasattr(self, "sliderArtifactThreshold") else 30
                mask = diff > threshold
                original_rgb = input_img.convert("RGB")
                orig_arr = np.array(original_rgb)
                overlay = orig_arr.copy()
                overlay[mask] = [255, 0, 255]
                alpha = 0.5
                result = np.uint8(alpha * overlay + (1 - alpha) * orig_arr)
                return Image.fromarray(result)
            elif mode == "Luminance":
                rgb = input_img.convert("RGB")
                arr = np.array(rgb).astype(np.float32)
                lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
                return Image.fromarray(np.uint8(lum))
            elif mode == "Contrast Map":
                gray = ImageOps.grayscale(input_img)
                arr = np.array(gray).astype(np.float32)
                window = 7
                mean_local = uniform_filter(arr, size=window)
                mean_sq_local = uniform_filter(arr**2, size=window)
                var_local = mean_sq_local - mean_local**2
                contrast = np.sqrt(np.maximum(var_local, 0))
                contrast = np.clip(contrast, 0, 255)
                return Image.fromarray(np.uint8(contrast))
            elif mode == "Edge Orientation Map":
                gray = ImageOps.grayscale(input_img)
                arr = np.array(gray).astype(np.float32)
                gy, gx = np.gradient(arr)
                orientation = np.degrees(np.arctan2(gy, gx))
                orientation_norm = (orientation + 180) / 360.0 * 255
                orientation_norm = np.clip(orientation_norm, 0, 255)
                return Image.fromarray(np.uint8(orientation_norm))
            elif mode == "LBP Analysis":
                if local_binary_pattern is None:
                    raise Exception("scikit-image not available")
                gray = ImageOps.grayscale(input_img)
                arr = np.array(gray)
                P = self.sliderLBPNeighbors.value() if hasattr(self, "sliderLBPNeighbors") else 8
                R = self.sliderLBPRadius.value() if hasattr(self, "sliderLBPRadius") else 1
                lbp = local_binary_pattern(arr, P, R, method='uniform')
                lbp_norm = 255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-6)
                return Image.fromarray(np.uint8(lbp_norm))
            elif mode == "Saturation Map":
                hsv = input_img.convert("HSV")
                hsv_arr = np.array(hsv)
                return Image.fromarray(hsv_arr[:,:,1])
            elif mode == "DCT Analysis":
                gray = ImageOps.grayscale(input_img)
                arr = np.array(gray).astype(np.float32)
                block_size = self.sliderDCTBlockSize.value() if hasattr(self, "sliderDCTBlockSize") else 16
                h, w = arr.shape
                h_c = h - (h % block_size)
                w_c = w - (w % block_size)
                img_cropped = arr[:h_c, :w_c]
                blocks = img_cropped.reshape(h_c // block_size, block_size, w_c // block_size, block_size)
                blocks = blocks.swapaxes(1, 2)
                dct_energy = np.zeros((h_c // block_size, w_c // block_size))
                for i in range(dct_energy.shape[0]):
                    for j in range(dct_energy.shape[1]):
                        block = blocks[i, j]
                        dct_block = dct(dct(block, norm='ortho', axis=0), norm='ortho', axis=1)
                        dct_energy[i, j] = np.sum(np.abs(dct_block))
                dct_energy = 255 * (dct_energy - dct_energy.min()) / (dct_energy.max() - dct_energy.min() + 1e-6)
                dct_img = Image.fromarray(np.uint8(dct_energy))
                dct_img = dct_img.resize(gray.size)
                return dct_img
            elif mode == "Gradient":
                gray = ImageOps.grayscale(input_img)
                arr = np.array(gray).astype(np.float32)
                gy, gx = np.gradient(arr)
                grad = np.sqrt(gx**2 + gy**2)
                grad = np.clip(grad, 0, 255)
                return Image.fromarray(np.uint8(grad))
            else:
                raise Exception(f"Mode '{mode}' is not supported in sequential analysis.")
        except Exception as e:
            raise Exception(f"Error processing mode '{mode}': {e}")

    # ------------------ Sequential Analysis Function ------------------
    def runSequentialAnalysis(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image file first.")
            return

        # Build pipeline 
        if self.radioPredefined.isChecked():
            pipeline_str = self.predefinedCombo.currentText()
            # Split on '->' and strip extra spaces
            selected_pipeline = [step.strip() for step in pipeline_str.split("->")]
        else:
            selected_pipeline = []
            for i in range(self.customPipelineList.count()):
                selected_pipeline.append(self.customPipelineList.item(i).text())

        debug_print(f"Running sequential analysis with pipeline: {selected_pipeline}")

        # Run the pipeline sequentially; each step's output is the next step's input
        current_img = self.image
        results = []
        titles = []
        for step in selected_pipeline:
            try:
                current_img = self.process_mode(step, current_img)
                results.append(current_img)
                titles.append(step)
            except Exception as e:
                supported = ["Original Image", "Grayscale", "Histogram Equalization", "Edge Detection",
                             "Fourier Transform", "Artifact Detection", "Luminance", "Contrast Map",
                             "Edge Orientation Map", "Saturation Map", "DCT Analysis", "Gradient"]
                QMessageBox.critical(self, "Sequential Analysis Error",
                                     f"Error in step '{step}': {e}\n\n"
                                     "Only chainable modes that produce an image output can be used in sequential analysis. "
                                     f"Supported modes are: {supported}")
                return

        # Display sequential results in a new figure with subplots
        n = len(results)
        seq_fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for i, (img, title) in enumerate(zip(results, titles)):
            if isinstance(img, Image.Image):
                axes[i].imshow(img)
            else:
                axes[i].imshow(np.array(img), cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        seq_fig.tight_layout()
        seq_fig.show()
        debug_print("Sequential analysis complete.")

    # ------------------ Individual Mode Visualization ------------------
    def visualizeImage(self):
        debug_print("Visualize Image button clicked")
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image file first.")
            return

        mode = self.visTypeCombo.currentText()
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        #Implement individual visualization.
        if mode == "Original Image":
            debug_print("Displaying original image")
            ax.imshow(self.image)
            ax.axis('off')
            ax.set_title("Original Image")
            self.current_data = np.array(self.image)

        elif mode == "Grayscale":
            debug_print("Converting image to grayscale")
            gray = ImageOps.grayscale(self.image)
            ax.imshow(gray, cmap='gray')
            ax.axis('off')
            ax.set_title("Grayscale")
            self.current_data = np.array(gray)

        elif mode == "Histogram":
            debug_print("Computing histogram")
            img_arr = np.array(self.image)
            if img_arr.ndim == 2:
                bins = int(self.histBinsEdit.text() or 256)
                ax.hist(img_arr.flatten(), bins=bins, color='gray')
            else:
                colors = ('red', 'green', 'blue')
                bins = int(self.histBinsEdit.text() or 256)
                for i, col in enumerate(colors):
                    hist, bin_edges = np.histogram(img_arr[:,:,i], bins=bins, range=(0,255))
                    ax.plot(bin_edges[:-1], hist, color=col, label=f'{col} channel')
                ax.legend()
            ax.set_title("Histogram")
            self.current_data = img_arr

        elif mode == "Edge Detection":
            debug_print("Applying edge detection")
            edges = self.image.filter(ImageFilter.FIND_EDGES)
            ax.imshow(edges)
            ax.axis('off')
            ax.set_title("Edge Detection")
            self.current_data = np.array(edges)

        elif mode == "Fourier Transform":
            debug_print("Computing Fourier transform")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            f_transform = np.fft.fft2(arr)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = 20 * np.log(np.abs(f_shift) + 1)
            ax.imshow(magnitude, cmap='gray')
            ax.axis('off')
            ax.set_title("Fourier Transform")
            self.current_data = magnitude

        elif mode == "Artifact Detection":
            debug_print("Performing artifact detection (bright pink)")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray).astype(np.float32)
            median_img = gray.filter(ImageFilter.MedianFilter(size=3))
            median_arr = np.array(median_img).astype(np.float32)
            diff = np.abs(arr - median_arr)
            threshold = int(self.histBinsEdit.text()) if False else self.sliderArtifactThreshold.value()
            # Use slider value if available
            threshold = self.sliderArtifactThreshold.value() if hasattr(self, "sliderArtifactThreshold") else 30
            mask = diff > threshold
            original_rgb = self.image.convert("RGB")
            orig_arr = np.array(original_rgb)
            overlay = orig_arr.copy()
            overlay[mask] = [255, 0, 255]
            alpha = 0.5
            result = np.uint8(alpha * overlay + (1 - alpha) * orig_arr)
            ax.imshow(result)
            ax.axis('off')
            ax.set_title("Artifact Detection (Bright Pink)")
            self.current_data = result

        elif mode in ["Error Level Analysis", "ELA"]:
            debug_print("Performing Error Level Analysis")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "ELA is best performed on JPEG images.")
            buf = io.BytesIO()
            #Fix for users loading non JPEG JPEG requires an image in "RGB" mode, you need to convert it before saving stupid I know >_<
            self.image.convert("RGB").save(buf, format="JPEG", quality=90)
            buf.seek(0)
            compressed = Image.open(buf)
            original = self.image.convert("RGB")
            compressed = compressed.convert("RGB")
            diff = ImageChops.difference(original, compressed)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            scale = 255.0 / max_diff if max_diff != 0 else 1
            ela_img = ImageEnhance.Brightness(diff).enhance(scale)
            ax.imshow(ela_img)
            ax.axis('off')
            ax.set_title("Error Level Analysis")
            self.current_data = np.array(ela_img)

        elif mode == "Noise Analysis":
            debug_print("Performing noise analysis")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray).astype(np.float32)
            blur_radius = self.sliderBlurRadius.value() if hasattr(self, "sliderBlurRadius") else 1
            blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_arr = np.array(blurred).astype(np.float32)
            noise = np.abs(arr - blurred_arr)
            ax.imshow(noise, cmap='inferno')
            ax.axis('off')
            ax.set_title("Noise Analysis")
            self.current_data = noise

        elif mode == "Level Sweep":
            debug_print("Performing level sweep")
            factors = [0.5, 0.75, 1.0, 1.25, 1.5]
            self.figure.clf()
            n = len(factors)
            for i, factor in enumerate(factors):
                ax_i = self.figure.add_subplot(1, n, i+1)
                enhancer = ImageEnhance.Brightness(self.image)
                adjusted = enhancer.enhance(factor)
                ax_i.imshow(adjusted)
                ax_i.axis('off')
                ax_i.set_title(f"x{factor}")
            self.current_data = np.array(self.image)
            self.canvas.draw()
            return

        elif mode == "Luminance":
            debug_print("Computing luminance")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb).astype(np.float32)
            lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
            ax.imshow(lum, cmap='gray')
            ax.axis('off')
            ax.set_title("Luminance")
            self.current_data = lum

        elif mode == "Gradient":
            debug_print("Computing gradient")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray).astype(np.float32)
            gy, gx = np.gradient(arr)
            grad = np.sqrt(gx**2 + gy**2)
            ax.imshow(grad, cmap='plasma')
            ax.axis('off')
            ax.set_title("Gradient")
            self.current_data = grad

        elif mode == "Principal Component Analysis":
            debug_print("Performing PCA")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb).astype(np.float32)
            h, w, c = arr.shape
            data = arr.reshape(-1, c)
            mean = np.mean(data, axis=0)
            centered = data - mean
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            pc = np.dot(centered, eigvecs[:, -1])
            pc_img = pc.reshape(h, w)
            ax.imshow(pc_img, cmap='viridis')
            ax.axis('off')
            ax.set_title("PCA - First Component")
            self.current_data = pc_img

        elif mode in ["Meta Data", "Metadata"]:
            debug_print("Extracting metadata")
            exif = self.image._getexif()
            if exif is None:
                meta_str = "No EXIF metadata found."
            else:
                from PIL.ExifTags import TAGS
                meta_list = [f"{TAGS.get(tag, tag)}: {value}" for tag, value in exif.items()]
                meta_str = "\n".join(meta_list)
            QMessageBox.information(self, "Metadata", meta_str)
            ax.axis('off')
            ax.set_title("Metadata (see message box)")
            self.current_data = None

        elif mode == "Geo Tags":
            debug_print("Extracting geo tags")
            exif = self.image._getexif()
            if exif is None:
                geo_str = "No EXIF metadata found."
            else:
                from PIL.ExifTags import TAGS, GPSTAGS
                gps = {}
                for tag, value in exif.items():
                    if TAGS.get(tag) == "GPSInfo":
                        for key in value:
                            gps[GPSTAGS.get(key, key)] = value[key]
                geo_str = "\n".join([f"{k}: {v}" for k, v in gps.items()]) if gps else "No GPS data found."
            QMessageBox.information(self, "Geo Tags", geo_str)
            ax.axis('off')
            ax.set_title("Geo Tags (see message box)")
            self.current_data = None

        elif mode == "Thumbnail Analysis":
            debug_print("Performing thumbnail analysis")
            exif = self.image.info.get("exif")
            if exif:
                QMessageBox.information(self, "Thumbnail Analysis", "Embedded thumbnail found (analysis not implemented).")
            else:
                QMessageBox.information(self, "Thumbnail Analysis", "No embedded thumbnail found.")
            ax.axis('off')
            ax.set_title("Thumbnail Analysis")
            self.current_data = None

        elif mode == "JPEG Analysis":
            debug_print("Performing JPEG analysis")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "JPEG Analysis is for JPEG images.")
                analysis = "Not a JPEG image."
            else:
                quant = getattr(self.image, "quantization", None)
                analysis = ("Quantization Tables:\n" + "\n".join([f"Table {k}: {v}" for k, v in quant.items()])
                            ) if quant else "No quantization tables found."
            QMessageBox.information(self, "JPEG Analysis", analysis)
            ax.axis('off')
            ax.set_title("JPEG Analysis (see message box)")
            self.current_data = None

        elif mode == "String Extraction":
            debug_print("Performing string extraction (message box)")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file available for extraction.")
                return
            try:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                min_len = int(self.stringMinLengthEdit.text() or 4)
                strs = re.findall(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}', data)
                strs = [s.decode('utf-8', errors='replace') for s in strs][:50]
                result = "\n".join(strs) if strs else "No strings found."
                QMessageBox.information(self, "String Extraction", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {e}")
            ax.axis('off')
            ax.set_title("String Extraction (see message box)")
            self.current_data = None

        elif mode == "Clone Detection":
            debug_print("Performing clone detection")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            block_size = self.sliderCloneBlockSize.value() if hasattr(self, "sliderCloneBlockSize") else 32
            h, w, _ = arr.shape
            blocks = {}
            clone_mask = np.zeros((h, w), dtype=bool)
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = arr[y:y+block_size, x:x+block_size]
                    hsh = hashlib.md5(block.tobytes()).hexdigest()
                    blocks.setdefault(hsh, []).append((x, y))
            for positions in blocks.values():
                if len(positions) > 1:
                    for (x, y) in positions:
                        clone_mask[y:y+block_size, x:x+block_size] = True
            overlay = arr.copy()
            overlay[clone_mask] = [255, 0, 0]
            alpha = 0.5
            result = np.uint8(alpha * overlay + (1 - alpha) * arr)
            ax.imshow(result)
            ax.axis('off')
            ax.set_title("Clone Detection (Red blocks)")
            self.current_data = result

        elif mode == "Digest":
            debug_print("Computing file digest")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file loaded.")
            else:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                digest = hashlib.sha256(data).hexdigest()
                QMessageBox.information(self, "Digest", f"SHA-256: {digest}")
            ax.axis('off')
            ax.set_title("Digest (see message box)")
            self.current_data = None

        elif mode == "Hidden Pixels":
            debug_print("Extracting hidden pixels (LSB)")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            lsb = (arr % 2) * 255
            ax.imshow(lsb, cmap='gray')
            ax.axis('off')
            ax.set_title("Hidden Pixels (LSB)")
            self.current_data = lsb

        elif mode == "ICC+":
            debug_print("Extracting ICC profile")
            icc = self.image.info.get("icc_profile", None)
            if icc:
                icc_bytes = icc.encode('latin1') if isinstance(icc, str) else icc
                icc_digest = hashlib.sha256(icc_bytes).hexdigest()
                info = f"ICC profile found, {len(icc_bytes)} bytes\nSHA-256: {icc_digest}"
            else:
                info = "No ICC profile found."
            QMessageBox.information(self, "ICC+ Analysis", info)
            ax.axis('off')
            ax.set_title("ICC+ (see message box)")
            self.current_data = None

        elif mode == "JPEG %":
            debug_print("Computing JPEG compression ratio")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "JPEG % is for JPEG images.")
                ax.axis('off')
                ax.set_title("JPEG % (Not applicable)")
                self.current_data = None
            else:
                size_file = os.path.getsize(self.image_file)
                w, h = self.image.size
                uncompressed = w * h * 3
                ratio = (size_file / uncompressed) * 100
                msg = f"Compression Ratio: {ratio:.2f}% of uncompressed size"
                QMessageBox.information(self, "JPEG % Analysis", msg)
                ax.axis('off')
                ax.set_title("JPEG % (see message box)")
                self.current_data = None

        elif mode == "String Extraction":
            debug_print("Performing string extraction (message box)")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file available for extraction.")
                return
            try:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                min_len = int(self.stringMinLengthEdit.text() or 4)
                strs = re.findall(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}', data)
                strs = [s.decode('utf-8', errors='replace') for s in strs][:50]
                result = "\n".join(strs) if strs else "No strings found."
                QMessageBox.information(self, "String Extraction", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {e}")
            ax.axis('off')
            ax.set_title("String Extraction (see message box)")
            self.current_data = None

        elif mode == "Clone Detection":
            debug_print("Performing clone detection")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            block_size = self.sliderCloneBlockSize.value() if hasattr(self, "sliderCloneBlockSize") else 32
            h, w, _ = arr.shape
            blocks = {}
            clone_mask = np.zeros((h, w), dtype=bool)
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = arr[y:y+block_size, x:x+block_size]
                    hsh = hashlib.md5(block.tobytes()).hexdigest()
                    blocks.setdefault(hsh, []).append((x, y))
            for positions in blocks.values():
                if len(positions) > 1:
                    for (x, y) in positions:
                        clone_mask[y:y+block_size, x:x+block_size] = True
            overlay = arr.copy()
            overlay[clone_mask] = [255, 0, 0]
            alpha = 0.5
            result = np.uint8(alpha * overlay + (1 - alpha) * arr)
            ax.imshow(result)
            ax.axis('off')
            ax.set_title("Clone Detection (Red blocks)")
            self.current_data = result

        elif mode == "Digest":
            debug_print("Computing file digest")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file loaded.")
            else:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                digest = hashlib.sha256(data).hexdigest()
                QMessageBox.information(self, "Digest", f"SHA-256: {digest}")
            ax.axis('off')
            ax.set_title("Digest (see message box)")
            self.current_data = None

        elif mode == "Hidden Pixels":
            debug_print("Extracting hidden pixels (LSB)")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            lsb = (arr % 2) * 255
            ax.imshow(lsb, cmap='gray')
            ax.axis('off')
            ax.set_title("Hidden Pixels (LSB)")
            self.current_data = lsb

        elif mode == "ICC+":
            debug_print("Extracting ICC profile")
            icc = self.image.info.get("icc_profile", None)
            if icc:
                icc_bytes = icc.encode('latin1') if isinstance(icc, str) else icc
                icc_digest = hashlib.sha256(icc_bytes).hexdigest()
                info = f"ICC profile found, {len(icc_bytes)} bytes\nSHA-256: {icc_digest}"
            else:
                info = "No ICC profile found."
            QMessageBox.information(self, "ICC+ Analysis", info)
            ax.axis('off')
            ax.set_title("ICC+ (see message box)")
            self.current_data = None

        elif mode == "JPEG %":
            debug_print("Computing JPEG compression ratio")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "JPEG % is for JPEG images.")
                ax.axis('off')
                ax.set_title("JPEG % (Not applicable)")
                self.current_data = None
            else:
                size_file = os.path.getsize(self.image_file)
                w, h = self.image.size
                uncompressed = w * h * 3
                ratio = (size_file / uncompressed) * 100
                msg = f"Compression Ratio: {ratio:.2f}% of uncompressed size"
                QMessageBox.information(self, "JPEG % Analysis", msg)
                ax.axis('off')
                ax.set_title("JPEG % (see message box)")
                self.current_data = None

        elif mode == "String Extraction":
            debug_print("Performing string extraction (message box)")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file available for extraction.")
                return
            try:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                min_len = int(self.stringMinLengthEdit.text() or 4)
                strs = re.findall(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}', data)
                strs = [s.decode('utf-8', errors='replace') for s in strs][:50]
                result = "\n".join(strs) if strs else "No strings found."
                QMessageBox.information(self, "String Extraction", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {e}")
            ax.axis('off')
            ax.set_title("String Extraction (see message box)")
            self.current_data = None

        elif mode == "Clone Detection":
            debug_print("Performing clone detection")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            block_size = self.sliderCloneBlockSize.value() if hasattr(self, "sliderCloneBlockSize") else 32
            h, w, _ = arr.shape
            blocks = {}
            clone_mask = np.zeros((h, w), dtype=bool)
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = arr[y:y+block_size, x:x+block_size]
                    hsh = hashlib.md5(block.tobytes()).hexdigest()
                    blocks.setdefault(hsh, []).append((x, y))
            for positions in blocks.values():
                if len(positions) > 1:
                    for (x, y) in positions:
                        clone_mask[y:y+block_size, x:x+block_size] = True
            overlay = arr.copy()
            overlay[clone_mask] = [255, 0, 0]
            alpha = 0.5
            result = np.uint8(alpha * overlay + (1 - alpha) * arr)
            ax.imshow(result)
            ax.axis('off')
            ax.set_title("Clone Detection (Red blocks)")
            self.current_data = result

        elif mode == "Digest":
            debug_print("Computing file digest")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file loaded.")
            else:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                digest = hashlib.sha256(data).hexdigest()
                QMessageBox.information(self, "Digest", f"SHA-256: {digest}")
            ax.axis('off')
            ax.set_title("Digest (see message box)")
            self.current_data = None

        elif mode == "Hidden Pixels":
            debug_print("Extracting hidden pixels (LSB)")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            lsb = (arr % 2) * 255
            ax.imshow(lsb, cmap='gray')
            ax.axis('off')
            ax.set_title("Hidden Pixels (LSB)")
            self.current_data = lsb

        elif mode == "ICC+":
            debug_print("Extracting ICC profile")
            icc = self.image.info.get("icc_profile", None)
            if icc:
                icc_bytes = icc.encode('latin1') if isinstance(icc, str) else icc
                icc_digest = hashlib.sha256(icc_bytes).hexdigest()
                info = f"ICC profile found, {len(icc_bytes)} bytes\nSHA-256: {icc_digest}"
            else:
                info = "No ICC profile found."
            QMessageBox.information(self, "ICC+ Analysis", info)
            ax.axis('off')
            ax.set_title("ICC+ (see message box)")
            self.current_data = None

        elif mode == "JPEG %":
            debug_print("Computing JPEG compression ratio")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "JPEG % is for JPEG images.")
                ax.axis('off')
                ax.set_title("JPEG % (Not applicable)")
                self.current_data = None
            else:
                size_file = os.path.getsize(self.image_file)
                w, h = self.image.size
                uncompressed = w * h * 3
                ratio = (size_file / uncompressed) * 100
                msg = f"Compression Ratio: {ratio:.2f}% of uncompressed size"
                QMessageBox.information(self, "JPEG % Analysis", msg)
                ax.axis('off')
                ax.set_title("JPEG % (see message box)")
                self.current_data = None

        elif mode == "String Extraction":
            debug_print("Performing string extraction (message box)")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file available for extraction.")
                return
            try:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                min_len = int(self.stringMinLengthEdit.text() or 4)
                strs = re.findall(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}', data)
                strs = [s.decode('utf-8', errors='replace') for s in strs][:50]
                result = "\n".join(strs) if strs else "No strings found."
                QMessageBox.information(self, "String Extraction", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {e}")
            ax.axis('off')
            ax.set_title("String Extraction (see message box)")
            self.current_data = None

        elif mode == "Clone Detection":
            debug_print("Performing clone detection")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            block_size = self.sliderCloneBlockSize.value() if hasattr(self, "sliderCloneBlockSize") else 32
            h, w, _ = arr.shape
            blocks = {}
            clone_mask = np.zeros((h, w), dtype=bool)
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = arr[y:y+block_size, x:x+block_size]
                    hsh = hashlib.md5(block.tobytes()).hexdigest()
                    blocks.setdefault(hsh, []).append((x, y))
            for positions in blocks.values():
                if len(positions) > 1:
                    for (x, y) in positions:
                        clone_mask[y:y+block_size, x:x+block_size] = True
            overlay = arr.copy()
            overlay[clone_mask] = [255, 0, 0]
            alpha = 0.5
            result = np.uint8(alpha * overlay + (1 - alpha) * arr)
            ax.imshow(result)
            ax.axis('off')
            ax.set_title("Clone Detection (Red blocks)")
            self.current_data = result

        elif mode == "Digest":
            debug_print("Computing file digest")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file loaded.")
            else:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                digest = hashlib.sha256(data).hexdigest()
                QMessageBox.information(self, "Digest", f"SHA-256: {digest}")
            ax.axis('off')
            ax.set_title("Digest (see message box)")
            self.current_data = None

        elif mode == "Hidden Pixels":
            debug_print("Extracting hidden pixels (LSB)")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            lsb = (arr % 2) * 255
            ax.imshow(lsb, cmap='gray')
            ax.axis('off')
            ax.set_title("Hidden Pixels (LSB)")
            self.current_data = lsb

        elif mode == "ICC+":
            debug_print("Extracting ICC profile")
            icc = self.image.info.get("icc_profile", None)
            if icc:
                icc_bytes = icc.encode('latin1') if isinstance(icc, str) else icc
                icc_digest = hashlib.sha256(icc_bytes).hexdigest()
                info = f"ICC profile found, {len(icc_bytes)} bytes\nSHA-256: {icc_digest}"
            else:
                info = "No ICC profile found."
            QMessageBox.information(self, "ICC+ Analysis", info)
            ax.axis('off')
            ax.set_title("ICC+ (see message box)")
            self.current_data = None

        elif mode == "JPEG %":
            debug_print("Computing JPEG compression ratio")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "JPEG % is for JPEG images.")
                ax.axis('off')
                ax.set_title("JPEG % (Not applicable)")
                self.current_data = None
            else:
                size_file = os.path.getsize(self.image_file)
                w, h = self.image.size
                uncompressed = w * h * 3
                ratio = (size_file / uncompressed) * 100
                msg = f"Compression Ratio: {ratio:.2f}% of uncompressed size"
                QMessageBox.information(self, "JPEG % Analysis", msg)
                ax.axis('off')
                ax.set_title("JPEG % (see message box)")
                self.current_data = None

        elif mode == "String Extraction":
            debug_print("Performing string extraction (message box)")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file available for extraction.")
                return
            try:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                min_len = int(self.stringMinLengthEdit.text() or 4)
                strs = re.findall(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}', data)
                strs = [s.decode('utf-8', errors='replace') for s in strs][:50]
                result = "\n".join(strs) if strs else "No strings found."
                QMessageBox.information(self, "String Extraction", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {e}")
            ax.axis('off')
            ax.set_title("String Extraction (see message box)")
            self.current_data = None

        elif mode == "Clone Detection":
            debug_print("Performing clone detection")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            block_size = self.sliderCloneBlockSize.value() if hasattr(self, "sliderCloneBlockSize") else 32
            h, w, _ = arr.shape
            blocks = {}
            clone_mask = np.zeros((h, w), dtype=bool)
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = arr[y:y+block_size, x:x+block_size]
                    hsh = hashlib.md5(block.tobytes()).hexdigest()
                    blocks.setdefault(hsh, []).append((x, y))
            for positions in blocks.values():
                if len(positions) > 1:
                    for (x, y) in positions:
                        clone_mask[y:y+block_size, x:x+block_size] = True
            overlay = arr.copy()
            overlay[clone_mask] = [255, 0, 0]
            alpha = 0.5
            result = np.uint8(alpha * overlay + (1 - alpha) * arr)
            ax.imshow(result)
            ax.axis('off')
            ax.set_title("Clone Detection (Red blocks)")
            self.current_data = result

        elif mode == "Digest":
            debug_print("Computing file digest")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file loaded.")
            else:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                digest = hashlib.sha256(data).hexdigest()
                QMessageBox.information(self, "Digest", f"SHA-256: {digest}")
            ax.axis('off')
            ax.set_title("Digest (see message box)")
            self.current_data = None

        elif mode == "Hidden Pixels":
            debug_print("Extracting hidden pixels (LSB)")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            lsb = (arr % 2) * 255
            ax.imshow(lsb, cmap='gray')
            ax.axis('off')
            ax.set_title("Hidden Pixels (LSB)")
            self.current_data = lsb

        elif mode == "ICC+":
            debug_print("Extracting ICC profile")
            icc = self.image.info.get("icc_profile", None)
            if icc:
                icc_bytes = icc.encode('latin1') if isinstance(icc, str) else icc
                icc_digest = hashlib.sha256(icc_bytes).hexdigest()
                info = f"ICC profile found, {len(icc_bytes)} bytes\nSHA-256: {icc_digest}"
            else:
                info = "No ICC profile found."
            QMessageBox.information(self, "ICC+ Analysis", info)
            ax.axis('off')
            ax.set_title("ICC+ (see message box)")
            self.current_data = None

        elif mode == "JPEG %":
            debug_print("Computing JPEG compression ratio")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "JPEG % is for JPEG images.")
                ax.axis('off')
                ax.set_title("JPEG % (Not applicable)")
                self.current_data = None
            else:
                size_file = os.path.getsize(self.image_file)
                w, h = self.image.size
                uncompressed = w * h * 3
                ratio = (size_file / uncompressed) * 100
                msg = f"Compression Ratio: {ratio:.2f}% of uncompressed size"
                QMessageBox.information(self, "JPEG % Analysis", msg)
                ax.axis('off')
                ax.set_title("JPEG % (see message box)")
                self.current_data = None

        elif mode == "String Extraction":
            debug_print("Performing string extraction (message box)")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file available for extraction.")
                return
            try:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                min_len = int(self.stringMinLengthEdit.text() or 4)
                strs = re.findall(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}', data)
                strs = [s.decode('utf-8', errors='replace') for s in strs][:50]
                result = "\n".join(strs) if strs else "No strings found."
                QMessageBox.information(self, "String Extraction", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {e}")
            ax.axis('off')
            ax.set_title("String Extraction (see message box)")
            self.current_data = None

        elif mode == "Clone Detection":
            debug_print("Performing clone detection")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            block_size = self.sliderCloneBlockSize.value() if hasattr(self, "sliderCloneBlockSize") else 32
            h, w, _ = arr.shape
            blocks = {}
            clone_mask = np.zeros((h, w), dtype=bool)
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = arr[y:y+block_size, x:x+block_size]
                    hsh = hashlib.md5(block.tobytes()).hexdigest()
                    blocks.setdefault(hsh, []).append((x, y))
            for positions in blocks.values():
                if len(positions) > 1:
                    for (x, y) in positions:
                        clone_mask[y:y+block_size, x:x+block_size] = True
            overlay = arr.copy()
            overlay[clone_mask] = [255, 0, 0]
            alpha = 0.5
            result = np.uint8(alpha * overlay + (1 - alpha) * arr)
            ax.imshow(result)
            ax.axis('off')
            ax.set_title("Clone Detection (Red blocks)")
            self.current_data = result

        elif mode == "Digest":
            debug_print("Computing file digest")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file loaded.")
            else:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                digest = hashlib.sha256(data).hexdigest()
                QMessageBox.information(self, "Digest", f"SHA-256: {digest}")
            ax.axis('off')
            ax.set_title("Digest (see message box)")
            self.current_data = None

        elif mode == "Hidden Pixels":
            debug_print("Extracting hidden pixels (LSB)")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            lsb = (arr % 2) * 255
            ax.imshow(lsb, cmap='gray')
            ax.axis('off')
            ax.set_title("Hidden Pixels (LSB)")
            self.current_data = lsb

        elif mode == "ICC+":
            debug_print("Extracting ICC profile")
            icc = self.image.info.get("icc_profile", None)
            if icc:
                icc_bytes = icc.encode('latin1') if isinstance(icc, str) else icc
                icc_digest = hashlib.sha256(icc_bytes).hexdigest()
                info = f"ICC profile found, {len(icc_bytes)} bytes\nSHA-256: {icc_digest}"
            else:
                info = "No ICC profile found."
            QMessageBox.information(self, "ICC+ Analysis", info)
            ax.axis('off')
            ax.set_title("ICC+ (see message box)")
            self.current_data = None

        elif mode == "JPEG %":
            debug_print("Computing JPEG compression ratio")
            if self.image.format != "JPEG":
                QMessageBox.warning(self, "Warning", "JPEG % is for JPEG images.")
                ax.axis('off')
                ax.set_title("JPEG % (Not applicable)")
                self.current_data = None
            else:
                size_file = os.path.getsize(self.image_file)
                w, h = self.image.size
                uncompressed = w * h * 3
                ratio = (size_file / uncompressed) * 100
                msg = f"Compression Ratio: {ratio:.2f}% of uncompressed size"
                QMessageBox.information(self, "JPEG % Analysis", msg)
                ax.axis('off')
                ax.set_title("JPEG % (see message box)")
                self.current_data = None

        elif mode == "String Extraction":
            debug_print("Performing string extraction (message box)")
            if not self.image_file:
                QMessageBox.warning(self, "Warning", "No file available for extraction.")
                return
            try:
                with open(self.image_file, "rb") as f:
                    data = f.read()
                min_len = int(self.stringMinLengthEdit.text() or 4)
                strs = re.findall(rb'[\x20-\x7E]{' + str(min_len).encode() + rb',}', data)
                strs = [s.decode('utf-8', errors='replace') for s in strs][:50]
                result = "\n".join(strs) if strs else "No strings found."
                QMessageBox.information(self, "String Extraction", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {e}")
            ax.axis('off')
            ax.set_title("String Extraction (see message box)")
            self.current_data = None

        # New additional modes
        elif mode == "Color Channel Analysis":
            debug_print("Performing color channel analysis")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            self.figure.clf()
            ax1 = self.figure.add_subplot(131)
            ax1.imshow(arr[:,:,0], cmap='Reds')
            ax1.axis('off')
            ax1.set_title("Red Channel")
            ax2 = self.figure.add_subplot(132)
            ax2.imshow(arr[:,:,1], cmap='Greens')
            ax2.axis('off')
            ax2.set_title("Green Channel")
            ax3 = self.figure.add_subplot(133)
            ax3.imshow(arr[:,:,2], cmap='Blues')
            ax3.axis('off')
            ax3.set_title("Blue Channel")
            self.current_data = arr
            self.canvas.draw()
            return

        elif mode == "Saturation Map":
            debug_print("Computing saturation map")
            hsv = self.image.convert("HSV")
            hsv_arr = np.array(hsv)
            sat = hsv_arr[:,:,1]
            ax.imshow(sat, cmap='viridis')
            ax.axis('off')
            ax.set_title("Saturation Map")
            self.current_data = sat

        elif mode == "Image Entropy":
            debug_print("Computing image entropy")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray)
            hist, _ = np.histogram(arr, bins=256, range=(0,255))
            p = hist / np.sum(hist)
            p = p[p > 0]
            entropy = -np.sum(p * np.log2(p))
            ax.imshow(arr, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Image Entropy: {entropy:.2f} bits")
            self.current_data = arr

        elif mode == "Histogram Equalization":
            debug_print("Performing histogram equalization")
            eq = ImageOps.equalize(self.image.convert("L"))
            ax.imshow(eq, cmap='gray')
            ax.axis('off')
            ax.set_title("Histogram Equalization")
            self.current_data = np.array(eq)

        elif mode == "Dominant Color Extraction":
            debug_print("Extracting dominant color")
            rgb = self.image.convert("RGB")
            arr = np.array(rgb)
            pixels = arr.reshape(-1, 3)
            quantized = (pixels // 16) * 16
            colors, counts = np.unique(quantized, axis=0, return_counts=True)
            dominant = colors[np.argmax(counts)]
            swatch = np.ones((100, 100, 3), dtype=np.uint8) * dominant
            ax.imshow(swatch)
            ax.axis('off')
            ax.set_title(f"Dominant Color: {dominant}")
            self.current_data = swatch

        elif mode == "Contrast Map":
            debug_print("Computing contrast map")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray).astype(np.float32)
            window = 7
            mean_local = uniform_filter(arr, size=window)
            mean_sq_local = uniform_filter(arr**2, size=window)
            var_local = mean_sq_local - mean_local**2
            contrast = np.sqrt(np.maximum(var_local, 0))
            contrast = np.clip(contrast, 0, 255)
            ax.imshow(contrast, cmap='gray')
            ax.axis('off')
            ax.set_title("Contrast Map")
            self.current_data = contrast

        elif mode == "Edge Orientation Map":
            debug_print("Computing edge orientation map")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray).astype(np.float32)
            gy, gx = np.gradient(arr)
            orientation = np.degrees(np.arctan2(gy, gx))
            orientation_norm = (orientation + 180) / 360.0
            ax.imshow(orientation_norm, cmap='hsv')
            ax.axis('off')
            ax.set_title("Edge Orientation Map")
            self.current_data = orientation_norm

        elif mode == "DCT Analysis":
            debug_print("Performing DCT Analysis")
            gray = ImageOps.grayscale(self.image)
            arr = np.array(gray).astype(np.float32)
            block_size = self.sliderDCTBlockSize.value() if hasattr(self, "sliderDCTBlockSize") else 16
            h, w = arr.shape
            h_c = h - (h % block_size)
            w_c = w - (w % block_size)
            img_cropped = arr[:h_c, :w_c]
            blocks = img_cropped.reshape(h_c // block_size, block_size, w_c // block_size, block_size)
            blocks = blocks.swapaxes(1, 2)
            dct_energy = np.zeros((h_c // block_size, w_c // block_size))
            for i in range(dct_energy.shape[0]):
                for j in range(dct_energy.shape[1]):
                    block = blocks[i, j]
                    dct_block = dct(dct(block, norm='ortho', axis=0), norm='ortho', axis=1)
                    dct_energy[i, j] = np.sum(np.abs(dct_block))
            dct_energy = 255 * (dct_energy - dct_energy.min()) / (dct_energy.max() - dct_energy.min() + 1e-6)
            dct_img = Image.fromarray(np.uint8(dct_energy))
            dct_img = dct_img.resize(gray.size)
            ax.imshow(dct_img, cmap='gray')
            ax.axis('off')
            ax.set_title("DCT Analysis")
            self.current_data = np.array(dct_img)
        else:
            QMessageBox.critical(self, "Error", "Unsupported mode selected in individual visualization.")
            return

        self.figure.tight_layout()
        self.canvas.draw()

    def popOutGraphs(self):
        debug_print("Pop Out Graphs button clicked")
        if self.current_data is None:
            QMessageBox.information(self, "Pop Out Graphs", "No visualization to pop out.")
            return
        fig, ax = plt.subplots()
        if self.current_data.ndim == 2:
            ax.imshow(self.current_data, cmap='gray')
        else:
            ax.imshow(self.current_data)
        ax.axis('off')
        ax.set_title(self.visTypeCombo.currentText())
        fig.show()
        debug_print("Graph popped out in a new window.")

    def saveVisualization(self):
        debug_print("Save Visualization button clicked")
        fname, _ = QFileDialog.getSaveFileName(self, "Save Visualization", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if fname:
            try:
                self.figure.savefig(fname)
                debug_print(f"Visualization saved to {fname}")
                QMessageBox.information(self, "Save Visualization", f"Visualization saved to:\n{fname}")
            except Exception as e:
                debug_print(f"Error saving visualization: {e}")
                QMessageBox.critical(self, "Error", f"Error saving visualization:\n{e}")

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = ImageVisualizer()
    window.resize(1000, 800)
    window.show()
    debug_print("ImageVisualizer started. Entering main loop.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
