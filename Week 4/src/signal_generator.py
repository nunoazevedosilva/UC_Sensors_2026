# import nidaqmx
# import numpy as np
# import time
# from nidaqmx.stream_writers import AnalogMultiChannelWriter
# from nidaqmx.constants import AcquisitionType
# from typing import Tuple

# class DAQSignalGenerator:
#     def __init__(self, device="Dev1", sampling_rate=10000, buffer_size=1000):
#         self.device = device
#         self.sampling_rate = sampling_rate
#         self.buffer_size = buffer_size
#         self.running = False
    
#     def generate_sine_wave(
#             self,
#             frequency: float,
#             amplitude: float,
#             sampling_rate: float,
#             buffer_size: int,
#             phase_in: float = 0.0,
#         ) -> Tuple[np.typing.NDArray[np.double], float]:
#         """Generates a sine wave with a specified phase."""
#         duration_time = buffer_size / sampling_rate
#         duration_radians = duration_time * 2 * np.pi
#         phase_out = (phase_in + duration_radians) % (2 * np.pi)
#         t = np.linspace(phase_in, phase_in + duration_radians, buffer_size, endpoint=False)
        
#         return (amplitude * np.sin(frequency * t), phase_out)

#     def send_signals(self, channels, frequencies, voltages):
#         """Continuously generates sine waves on multiple channels."""
#         with nidaqmx.Task() as task:
#             for channel in channels:
#                 task.ao_channels.add_ao_voltage_chan(f"{self.device}/{channel}")
            
#             task.timing.cfg_samp_clk_timing(self.sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
#             actual_sampling_rate = task.timing.samp_clk_rate
#             print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")
            
#             waveforms = []
#             for freq, volt in zip(frequencies, voltages):
#                 data, _ = self.generate_sine_wave(
#                     frequency=freq,
#                     amplitude=volt,
#                     sampling_rate=actual_sampling_rate,
#                     buffer_size=self.buffer_size,
#                 )
#                 waveforms.append(data)
            
#             writer = AnalogMultiChannelWriter(task.out_stream)
#             writer.write_many_sample(np.array(waveforms))
            
#             task.start()
#             input("Generating voltage continuously. Press Enter to stop.\n")
#             task.stop()


import nidaqmx
import numpy as np
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.constants import AcquisitionType
from typing import Tuple, List

class DAQSignalGenerator:
    def __init__(self, device="Dev1", sampling_rate=100000, num_cycles=1):
        """
        Initializes the DAQSignalGenerator.

        Parameters:
            device (str): NI device identifier.
            sampling_rate (int): Desired sampling rate (in samples/second).
            num_cycles (int): Number of full sine wave cycles per buffer.
                              The buffer size will be computed as:
                              buffer_size = ceil(sampling_rate/frequency * num_cycles)
        """
        self.device = device
        self.sampling_rate = sampling_rate
        self.num_cycles = num_cycles
        self.task = None  # DAQmx task

    def generate_sine_wave(self, frequency: float, amplitude: float, offset: float = 0.0, phase_in: float = 0.0):
        """
        Generates a sine wave for a given frequency, amplitude, and starting phase.

        The number of samples is computed dynamically so that the buffer contains
        an integer number of cycles.

        Parameters:
            frequency (float): Frequency of the sine wave in Hz.
            amplitude (float): Amplitude of the sine wave.
            phase_in (float): Starting phase (in radians).

        Returns:
            waveform (np.ndarray): Array containing the sine wave samples.
            phase_out (float): Phase value at the end of the buffer.
            buffer_size (int): Number of samples in the waveform.
        """
        if frequency != 0:

            samples_per_cycle = self.sampling_rate / frequency
        else:
            samples_per_cycle = self.sampling_rate
  
        buffer_size = int(np.ceil(samples_per_cycle * self.num_cycles))

        duration_time = buffer_size / self.sampling_rate

        total_phase_change = 2 * np.pi * frequency * duration_time
        phase_out = (phase_in + total_phase_change) % (2 * np.pi)

        t = np.linspace(0, duration_time, buffer_size, endpoint=False)
    
        waveform = amplitude * np.cos(2 * np.pi * frequency * t + phase_in) + offset
        return waveform, phase_out, buffer_size

    def start_signals(self, channels, frequencies, voltages, offset = 0.0):
        """
        Starts generating continuous sine wave signals on the specified channels.

        Parameters:
            channels (list of str): List of channel names (e.g., ['ao0', 'ao1']).
            frequencies (list of float): Frequencies for each channel.
            voltages (list of float): Amplitudes (voltage levels) for each channel.
        """

        if self.task:
            self.stop()

        self.task = nidaqmx.Task()

        for channel in channels:
            self.task.ao_channels.add_ao_voltage_chan(f"{self.device}/{channel}")


        print(self.sampling_rate)
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sampling_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )

        actual_sampling_rate = self.task.timing.samp_clk_rate
        print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")

        waveforms = []
        buffer_sizes = []
        for freq, volt in zip(frequencies, voltages):
            waveform, _, buf_size = self.generate_sine_wave(
                frequency=freq,
                amplitude=volt,
                phase_in=0.0, offset=offset 
            )
            waveforms.append(waveform)
            buffer_sizes.append(buf_size)

     
        common_buffer_size = max(buffer_sizes)

        for i in range(len(waveforms)):
            if len(waveforms[i]) < common_buffer_size:
                repeats = int(np.ceil(common_buffer_size / len(waveforms[i])))
                waveform_extended = np.tile(waveforms[i], repeats)[:common_buffer_size]
                waveforms[i] = waveform_extended

        writer = AnalogMultiChannelWriter(self.task.out_stream)
        writer.write_many_sample(np.array(waveforms))

        self.task.start()
        print("Signal generation started. Call the 'stop()' method to halt the output.")

    def stop(self):
        """Stops and closes the current DAQ task."""
        if self.task:
            self.task.stop()
            self.task.close()
            self.task = None
            print("Signal generation stopped.")


    def play_signal_from_array(self, channels: List[str], signals: List[np.ndarray], sampling_rates: List[int]):
        """Plays predefined signals from NumPy arrays on multiple channels."""
        if len(channels) != len(signals) or len(channels) != len(sampling_rates):
            raise ValueError("Each channel must have a corresponding signal and sampling rate.")

        if self.task:
            self.stop()

        self.task = nidaqmx.Task()

        for channel in channels:
            self.task.ao_channels.add_ao_voltage_chan(f"{self.device}/{channel}")

        max_sr = max(sampling_rates)
        self.task.timing.cfg_samp_clk_timing(
            max_sr, sample_mode=AcquisitionType.FINITE, samps_per_chan=max(len(sig) for sig in signals)
        )

        resampled_signals = []
        for sig, sr in zip(signals, sampling_rates):
            if sr != max_sr:
                new_length = int(len(sig) * (max_sr / sr))
                resampled_sig = np.interp(np.linspace(0, len(sig), new_length, endpoint=False), np.arange(len(sig)), sig)
                resampled_signals.append(resampled_sig)
            else:
                resampled_signals.append(sig)

        max_length = max(len(sig) for sig in resampled_signals)
        for i in range(len(resampled_signals)):
            if len(resampled_signals[i]) < max_length:
                resampled_signals[i] = np.pad(resampled_signals[i], (0, max_length - len(resampled_signals[i])))

        signals_array = np.vstack(resampled_signals)

        writer = AnalogMultiChannelWriter(self.task.out_stream)
        writer.write_many_sample(signals_array)

        self.task.start()
        print("Playing predefined signals...")
        self.task.wait_until_done()
        self.stop()
