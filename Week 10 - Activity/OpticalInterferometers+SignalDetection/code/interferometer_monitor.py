import os
import sys
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from scipy import signal
import sounddevice as sd


LOGO_PATH = os.path.join(os.path.dirname(__file__), "utils", "team_logo.png")

class SineGenerator:
    def __init__(self, fs=44100):
        self.fs = fs
        self.frequency = 440.0
        self.amplitude = 0.1
        self.is_running = False
        self.phase = 0
        self.stream = None

    def callback(self, outdata, frames, time, status):
        t = (self.phase + np.arange(frames)) / self.fs
        outdata[:, 0] = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        self.phase += frames

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.phase = 0
            self.stream = sd.OutputStream(channels=1, callback=self.callback, samplerate=self.fs)
            self.stream.start()

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()

class DAQThread(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, device="Dev1", rate=20000, chunk_size=1000):
        super().__init__()
        self.device = device
        self.rate = rate
        self.chunk_size = chunk_size
        self.running = False

    def run(self):
        self.running = True
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(f"{self.device}/ai0")
                task.timing.cfg_samp_clk_timing(rate=self.rate, 
                                                sample_mode=AcquisitionType.CONTINUOUS, 
                                                samps_per_chan=self.chunk_size)
                while self.running:
                    data = task.read(number_of_samples_per_channel=self.chunk_size)
                    self.data_signal.emit(np.array(data))
        except Exception as e:
            print(f"DAQ Error: {e}")
        finally:
            self.running = False

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.fs_daq = 20000
        self.max_duration = 5.0
        self.buffer_size = int(self.fs_daq * self.max_duration)
        self.ema_alpha = 0.3
        self.raw_data = np.zeros(self.buffer_size)
        self.avg_fft = None
        self.audio_gen = SineGenerator()

        # Window Config
        self.setWindowTitle(f"NI-DAQ Monitor - {self.device_name}")
        self.setWindowIcon(QtGui.QIcon(LOGO_PATH))
        self.resize(1400, 950)
        
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # Plots
        plot_layout = QtWidgets.QVBoxLayout()
        self.time_plot = pg.PlotWidget(title="Real-Time Signal")
        self.time_curve = self.time_plot.plot()
        self.fft_plot = pg.PlotWidget(title="EMA FFT (Peak Search > 1kHz)")
        self.fft_curve = self.fft_plot.plot()
        self.fft_plot.setLogMode(False, True)
        self.peak_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(QtGui.QColor(255, 0, 0, 120), width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.fft_plot.addItem(self.peak_line)
        plot_layout.addWidget(self.time_plot); plot_layout.addWidget(self.fft_plot)

        # Sidebar
        self.controls_group = QtWidgets.QGroupBox("Controls")
        sidebar = QtWidgets.QVBoxLayout(self.controls_group)

        # Add Logo
        self.logo_label = QtWidgets.QLabel()
        logo_pix = QtGui.QPixmap(LOGO_PATH)
        if not logo_pix.isNull():
            self.logo_label.setPixmap(logo_pix.scaledToWidth(200, QtCore.Qt.TransformationMode.SmoothTransformation))
            self.logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            sidebar.addWidget(self.logo_label)

        # Theme & DAQ
        self.theme_btn = QtWidgets.QPushButton("🌙 DARK MODE"); self.theme_btn.setCheckable(True)
        self.theme_btn.toggled.connect(self.toggle_theme)
        self.daq_btn = QtWidgets.QPushButton("START ACQUISITION"); self.daq_btn.setCheckable(True)
        self.daq_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; height: 40px;")
        self.daq_btn.toggled.connect(self.toggle_daq)
        sidebar.addWidget(self.theme_btn); sidebar.addWidget(self.daq_btn)
        sidebar.addWidget(self.line_separator())

        # Peak Marker Toggle
        self.peak_marker_enable = QtWidgets.QCheckBox("Show Peak Marker (> 1kHz)"); self.peak_marker_enable.setChecked(True)
        self.peak_marker_enable.toggled.connect(self.toggle_peak_marker)
        sidebar.addWidget(self.peak_marker_enable); sidebar.addWidget(self.line_separator())

        # Audio Gen
        sidebar.addWidget(QtWidgets.QLabel("<b>3.5mm Jack Output</b>"))
        self.audio_btn = QtWidgets.QPushButton("Start Sine Wave"); self.audio_btn.setCheckable(True)
        self.audio_btn.toggled.connect(self.toggle_audio)
        self.audio_freq_l, self.audio_freq_spin = self.create_spin(20, 20000, 440, "Output Freq (Hz)")
        self.audio_vol_l, self.audio_vol_spin = self.create_double_spin(0.0, 1.0, 0.1, "Volume")
        self.audio_freq_spin.valueChanged.connect(self.update_audio_params); self.audio_vol_spin.valueChanged.connect(self.update_audio_params)
        sidebar.addWidget(self.audio_btn); sidebar.addLayout(self.audio_freq_l); sidebar.addLayout(self.audio_vol_l)
        sidebar.addWidget(self.line_separator())

        # Filters
        sidebar.addWidget(QtWidgets.QLabel("<b>Filters</b>"))
        self.comb_enable = QtWidgets.QCheckBox("Comb Filter")
        self.comb_f0_l, self.comb_f0_spin = self.create_spin(10, 5000, 50, "Base Freq (Hz)")
        self.comb_q_l, self.comb_q_spin = self.create_double_spin(0.1, 100.0, 30.0, "Q Factor")
        sidebar.addWidget(self.comb_enable); sidebar.addLayout(self.comb_f0_l); sidebar.addLayout(self.comb_q_l)
        
        self.lp_enable = QtWidgets.QCheckBox("Low Pass")
        self.lp_f_l, self.lp_f_spin = self.create_spin(10, 9999, 1000, "Cutoff (Hz)")
        self.lp_order_l, self.lp_order_spin = self.create_spin(1, 12, 4, "Order")
        sidebar.addWidget(self.lp_enable); sidebar.addLayout(self.lp_f_l); sidebar.addLayout(self.lp_order_l)

        self.hp_enable = QtWidgets.QCheckBox("High Pass")
        self.hp_f_l, self.hp_f_spin = self.create_spin(1, 5000, 10, "Cutoff (Hz)")
        self.hp_order_l, self.hp_order_spin = self.create_spin(1, 12, 4, "Order")
        sidebar.addWidget(self.hp_enable); sidebar.addLayout(self.hp_f_l); sidebar.addLayout(self.hp_order_l)

        sidebar.addStretch()
        
        # Scale/Window
        self.autoscale_btn = QtWidgets.QPushButton("Autoscale: ON"); self.autoscale_btn.setCheckable(True); self.autoscale_btn.setChecked(True)
        self.window_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.window_slider.setRange(10, 5000); self.window_slider.setValue(1000)
        self.window_slider.valueChanged.connect(self.update_x_axis)
        self.slider_label = QtWidgets.QLabel("View Window: 1.00s")
        sidebar.addWidget(self.autoscale_btn); sidebar.addWidget(self.slider_label); sidebar.addWidget(self.window_slider)

        main_layout.addLayout(plot_layout, stretch=4); main_layout.addWidget(self.controls_group, stretch=1)

        self.thread = DAQThread(device=self.device_name, rate=self.fs_daq)
        self.thread.data_signal.connect(self.process_data)
        self.fft_timer = QtCore.QTimer(); self.fft_timer.timeout.connect(self.calculate_fft)
        
        self.toggle_theme(False); self.update_x_axis()

    def toggle_peak_marker(self, checked):
        self.peak_line.setVisible(checked)

    def toggle_theme(self, dark_mode):
        if dark_mode:
            self.theme_btn.setText("☀️ LIGHT MODE")
            self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
            self.controls_group.setStyleSheet("color: #ffffff;")
            bg, axis, sig, fft = '#1a1a1a', '#EEE', 'y', 'c'
            peak = QtGui.QColor(255, 50, 50, 120) # Red for Dark
        else:
            self.theme_btn.setText("🌙 DARK MODE")
            self.setStyleSheet("background-color: #f5f5f5; color: #000000;")
            self.controls_group.setStyleSheet("color: #000000;")
            bg, axis, sig, fft = '#ffffff', '#333', '#0055ff', '#cc0000'
            peak = QtGui.QColor(46, 204, 113, 120) # Green for White
        
        self.time_plot.setBackground(bg); self.fft_plot.setBackground(bg)
        self.time_curve.setPen(sig); self.fft_curve.setPen(fft)
        self.peak_line.setPen(pg.mkPen(peak, width=2, style=QtCore.Qt.PenStyle.DashLine))
        for p in [self.time_plot, self.fft_plot]:
            for ax in ['bottom', 'left']:
                p.getAxis(ax).setLabel(**{'color': axis}); p.getAxis(ax).setPen(axis)

    def toggle_daq(self, checked):
        if checked:
            self.daq_btn.setText("STOP ACQUISITION"); self.daq_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; height: 40px;")
            self.thread.start(); self.fft_timer.start(500)
        else:
            self.daq_btn.setText("START ACQUISITION"); self.daq_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; height: 40px;")
            self.thread.stop(); self.fft_timer.stop()

    def create_spin(self, min_v, max_v, def_v, label):
        l = QtWidgets.QHBoxLayout(); s = QtWidgets.QSpinBox(); s.setRange(min_v, max_v); s.setValue(def_v)
        l.addWidget(QtWidgets.QLabel(label)); l.addWidget(s); return l, s

    def create_double_spin(self, min_v, max_v, def_v, label):
        l = QtWidgets.QHBoxLayout(); s = QtWidgets.QDoubleSpinBox(); s.setRange(min_v, max_v); s.setValue(def_v); s.setSingleStep(0.05)
        l.addWidget(QtWidgets.QLabel(label)); l.addWidget(s); return l, s

    def line_separator(self):
        f = QtWidgets.QFrame(); f.setFrameShape(QtWidgets.QFrame.Shape.HLine); return f

    def toggle_audio(self, checked):
        if checked:
            self.audio_btn.setText("Stop Sine Wave"); self.audio_btn.setStyleSheet("background-color: #e74c3c; color: white;")
            self.update_audio_params(); self.audio_gen.start()
        else:
            self.audio_btn.setText("Start Sine Wave"); self.audio_btn.setStyleSheet(""); self.audio_gen.stop()

    def update_audio_params(self):
        self.audio_gen.frequency = self.audio_freq_spin.value(); self.audio_gen.amplitude = self.audio_vol_spin.value()

    def update_x_axis(self):
        duration = self.window_slider.value() / 1000.0
        self.slider_label.setText(f"View Window: {duration:.2f}s")
        self.time_plot.setXRange(0, duration, padding=0)

    def apply_filters(self, data):
        if len(data) < 100: return data
        f = data.copy()
        try:
            if self.comb_enable.isChecked():
                b, a = signal.iircomb(self.comb_f0_spin.value(), self.comb_q_spin.value(), ftype='notch', fs=self.fs_daq)
                f = signal.filtfilt(b, a, f)
            if self.lp_enable.isChecked():
                b, a = signal.butter(self.lp_order_spin.value(), self.lp_f_spin.value(), btype='low', fs=self.fs_daq)
                f = signal.filtfilt(b, a, f)
            if self.hp_enable.isChecked():
                b, a = signal.butter(self.hp_order_spin.value(), self.hp_f_spin.value(), btype='high', fs=self.fs_daq)
                f = signal.filtfilt(b, a, f)
        except: pass
        return f

    def process_data(self, new_data):
        self.raw_data = np.roll(self.raw_data, -len(new_data)); self.raw_data[-len(new_data):] = new_data
        dur = self.window_slider.value() / 1000.0; pts = int(dur * self.fs_daq)
        pad = 200; slice_size = min(pts + pad, self.buffer_size)
        v_data = self.raw_data[-slice_size:]; f_view = self.apply_filters(v_data)
        disp = f_view[-pts:]; self.time_curve.setData(np.linspace(0, dur, len(disp)), disp)
        if self.autoscale_btn.isChecked(): self.time_plot.autoRange()

    def calculate_fft(self):
        slen = int(0.5 * self.fs_daq); d = self.apply_filters(self.raw_data[-slen:])
        n = len(d); fr = np.fft.rfftfreq(n, d=1/self.fs_daq); mg = np.abs(np.fft.rfft(d)) / n + 1e-9
        if self.avg_fft is None or len(self.avg_fft) != len(mg): self.avg_fft = mg
        else: self.avg_fft = (self.ema_alpha * mg) + (1 - self.ema_alpha) * self.avg_fft
        self.fft_curve.setData(fr, self.avg_fft)
        m = fr >= 1000
        if np.any(m):
            p_mg = self.avg_fft[m]; p_fr = fr[m]; idx = np.argmax(p_mg); p_f = p_fr[idx]
            self.peak_line.setValue(p_f); self.fft_plot.setTitle(f"EMA FFT (Peak > 1kHz: {p_f:.1f} Hz)")
        if self.autoscale_btn.isChecked(): self.fft_plot.autoRange()

    def closeEvent(self, e):
        self.audio_gen.stop(); self.thread.stop(); e.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dev, ok = QtWidgets.QInputDialog.getText(None, "Device Selection", "NI Device ID (e.g. Dev1):", QtWidgets.QLineEdit.EchoMode.Normal, "Dev1")
    if ok and dev:
        window = MainWindow(dev); window.show(); sys.exit(app.exec())
    else: sys.exit()