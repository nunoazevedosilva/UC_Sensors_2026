import nidaqmx
import numpy as np
import time
import math
import os

from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogSingleChannelReader
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from typing import List, Tuple 

class NIDAQ:
    def __init__(self, device = "Dev1", sampling_rate = 100000, num_cycles = 1):
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
        self.task = None

    def generate_sine_wave(self, frequency, amplitude, offset = 0.0, phase_in = 0.0):
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
        if frequency == 0:
            buffer_size = int(self.sampling_rate * self.num_cycles)
            t = np.arange(buffer_size) / self.sampling_rate
            waveform = amplitude * np.ones(buffer_size) * np.cos(phase_in) + offset
            phase_out = phase_in
            return waveform, phase_out, buffer_size

        buffer_size = int(self.sampling_rate * self.num_cycles)

        step = 2 * np.pi * frequency / self.sampling_rate
        n = np.arange(buffer_size)
        phase = phase_in + step * n
        waveform = amplitude * np.cos(phase) + offset

        # final phase to allow continuity
        phase_out = (phase_in + step * buffer_size) % (2 * np.pi)

        return waveform, phase_out, buffer_size

    def start_signals(self, channels, frequencies, voltages, offset = 0.0, silence = False):
        if self.task:
            self.stop()

        self.task = nidaqmx.Task()

        for ch in channels:
            self.task.ao_channels.add_ao_voltage_chan(f"{self.device}/{ch}")

        base_waves = []
        base_lengths = []
        for f, v in zip(frequencies, voltages):
            wave, _, n = self.generate_sine_wave(frequency = f, amplitude = v, offset = offset, phase_in = 0.0)
            base_waves.append(wave)        # length n, integer cycles of that f
            base_lengths.append(len(wave)) # samples/chan

        def lcm(a, b):
            return a * b // math.gcd(a, b)

        common_len = base_lengths[0]
        for n in base_lengths[1:]:
            common_len = lcm(common_len, n)

        waves = []
        for wave, n in zip(base_waves, base_lengths):
            reps = common_len // n
            waves.append(np.tile(wave, reps))

        self.task.timing.cfg_samp_clk_timing(
            rate = self.sampling_rate,
            sample_mode = AcquisitionType.CONTINUOUS,
            samps_per_chan = common_len
        )

        self.task.out_stream.output_buf_size = common_len


        writer = AnalogMultiChannelWriter(self.task.out_stream)
        data = np.vstack(waves)  # shape: (len(channels), common_len)
        writer.write_many_sample(data)

        self.task.start()
        if not silence:
            print(f"Actual sampling rate: {self.task.timing.samp_clk_rate:g} S/s")
            print(f"Buffer length per channel: {common_len} samples")
            print("Signal generation started. Call stop() to halt the output.")
            
            
    def start_frequency_comb(self, channels, comb_frequencies, comb_amplitudes, offsets = None, silence = False):
        """
        Starts continuous frequency-comb output on multiple channels.
        """
        if self.task:
            self.stop(silence=True)

        n_channels = len(channels)
        if len(comb_frequencies) != n_channels or len(comb_amplitudes) != n_channels:
            raise ValueError("comb_frequencies and comb_amplitudes must match number of channels")

        if offsets is None:
            offsets = [0.0] * n_channels

        # Create the task
        self.task = nidaqmx.Task()
        for ch in channels:
            self.task.ao_channels.add_ao_voltage_chan(f"{self.device}/{ch}")

        # --- Generate combs per channel ---
        n_samples = int(self.sampling_rate * self.num_cycles)
        t = np.arange(n_samples) / self.sampling_rate

        all_waves = []
        for freqs, amps, off in zip(comb_frequencies, comb_amplitudes, offsets):
            wave = np.zeros_like(t)
            for f, a in zip(freqs, amps):
                wave += a * np.cos(2 * np.pi * f * t)
            wave += off
            wave = np.clip(wave, -10.0, 10.0)
            all_waves.append(wave)

        data = np.vstack(all_waves)

        self.task.timing.cfg_samp_clk_timing(
            rate = self.sampling_rate,
            sample_mode = AcquisitionType.CONTINUOUS,
            samps_per_chan = n_samples
        )

        writer = AnalogMultiChannelWriter(self.task.out_stream)
        writer.write_many_sample(data)

        self.task.start()
        if not silence:
            print(f"Started continuous frequency comb output on {len(channels)} channel(s)")
            for i, ch in enumerate(channels):
                print(f"  {ch}: {comb_frequencies[i]} Hz")
            print(f"Sampling rate: {self.sampling_rate:g} S/s, buffer: {n_samples} samples per channel.")


    def stop(self, silence = False):
        """Stops and closes the current DAQ task."""
        if self.task:
            self.task.stop()
            self.task.close()
            self.task = None
            if not silence:
                print("Signal generation stopped.")


    def play_signal_from_array(self, channels, signals, sampling_rates):
        """
        Plays signals using software-timing for USB-6009 hardware.
        Input signals must be pre-scaled to 0V - 5V range.
        """
        if len(channels) != len(signals):
            raise ValueError("Each channel must have a corresponding signal.")

        # 1. Resample signals to a common rate
        max_sr = max(sampling_rates)
        resampled_signals = []
        for sig, sr in zip(signals, sampling_rates):
            if sr != max_sr:
                new_length = int(len(sig) * (max_sr / sr))
                resampled_sig = np.interp(
                    np.linspace(0, len(sig), new_length, endpoint=False), 
                    np.arange(len(sig)), 
                    sig
                )
                resampled_signals.append(resampled_sig)
            else:
                resampled_signals.append(sig)

        # 2. Synchronize lengths by padding with the last value (or 0)
        max_length = max(len(sig) for sig in resampled_signals)
        final_signals = []
        for sig in resampled_signals:
            if len(sig) < max_length:
                final_signals.append(np.pad(sig, (0, max_length - len(sig)), mode='edge'))
            else:
                final_signals.append(sig)

        # Prepare data for iteration: (samples, channels)
        data_to_write = np.vstack(final_signals).T 

        # 3. Execution Loop
        with nidaqmx.Task() as task:
            for channel in channels:
                # Hardware limits for USB-6009: 0 to 5V
                task.ao_channels.add_ao_voltage_chan(
                    f"{self.device}/{channel}", 
                    min_val=0.0, 
                    max_val=5.0
                )

            print(f"Playing {max_length} samples at {max_sr} Hz...")
            
            # Use perf_counter for microsecond-level timing
            start_time = time.perf_counter()
            for i in range(max_length):
                # Write current sample(s) - data_to_write[i] is a 1D array of size len(channels)
                task.write(data_to_write[i].tolist())
                
                # Precise wait for next sample
                target_time = start_time + (i + 1) / max_sr
                while time.perf_counter() < target_time:
                    pass 

        print("Playback finished.")

    def acquire_data_single_channel(self, duration = 30, channel = 'ai0', rate = 10000, buffer_size = 1000, silence = False):
        """
        Acquires data from a single analog input channel in differential mode over a specified duration,
        ensuring the total acquisition time is precise based on sample count.

        :param duration: Acquisition duration in seconds.
        :param channel: DAQ device analog input channel.
        :param rate: Sampling rate in samples per second.
        :param buffer_size: Number of samples to read per iteration.
        :param silence: If True, suppress console output.
        :return: NumPy array of acquired data.
        """
        total_samples = int(duration * rate)
        data = []
        samples_collected = 0

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.device}/{channel}",
                                                terminal_config = TerminalConfiguration.DIFF)

            task.timing.cfg_samp_clk_timing(rate = rate,
                                            sample_mode = AcquisitionType.CONTINUOUS,
                                            samps_per_chan = buffer_size)

            reader = AnalogSingleChannelReader(task.in_stream)
            buffer = np.zeros(buffer_size)

            if not silence:
                print(f"Starting data acquisition for {duration} seconds ({total_samples} samples).")

            task.start()

            try:
                while samples_collected < total_samples:
                    samples_to_read = min(buffer_size, total_samples - samples_collected)
                    reader.read_many_sample(buffer[:samples_to_read], number_of_samples_per_channel=samples_to_read, timeout=10.0)
                    data.append(buffer[:samples_to_read].copy())
                    samples_collected += samples_to_read

            except KeyboardInterrupt:
                print("Data acquisition interrupted by user.")

            finally:
                if not silence:
                    print(f"Data acquisition complete. Collected {samples_collected} samples.")

        return np.concatenate(data)
    
    def synchronized_output_input(self,
                                output_channel,
                                input_channels,
                                frequency,
                                amplitude,
                                offset,
                                duration,
                                silence = False):
        """
        Outputs a sine wave to the specified AO channel and simultaneously acquires data 
        from one or more AI channels using synchronized timing and a start trigger.

        Parameters:
            output_channel (str): AO channel (e.g., 'ao0')
            input_channels (List[str]): List of AI channels (e.g., ['ai0', 'ai1'])
            frequency (float): Frequency of the sine wave (Hz)
            amplitude (float): Amplitude of the sine wave
            offset (float): DC offset for the sine wave
            duration (float): Duration of signal generation and acquisition (seconds)
            silence (bool): If True, suppresses prints

        Returns:
            Tuple of (input_signals [channels x samples], output_waveform, time_array)
        """
        n_samples = int(duration * self.sampling_rate)
        time_array = np.linspace(0, duration, n_samples, endpoint=False)
        output_waveform = amplitude * np.sin(2 * np.pi * frequency * time_array) + offset

        n_channels = len(input_channels)
        input_buffer = np.zeros((n_channels, n_samples))

        with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
            ao_task.ao_channels.add_ao_voltage_chan(f"{self.device}/{output_channel}")
            ao_task.timing.cfg_samp_clk_timing(rate = self.sampling_rate,
                                            sample_mode = AcquisitionType.FINITE,
                                            samps_per_chan = n_samples)

            for ch in input_channels:
                ai_task.ai_channels.add_ai_voltage_chan(f"{self.device}/{ch}",
                                                        terminal_config=TerminalConfiguration.RSE,
                                                        min_val = -10.0,
                                                        max_val = 10.0)

            ai_task.timing.cfg_samp_clk_timing(rate = self.sampling_rate,
                                            source = f"/{self.device}/ao/SampleClock",
                                            sample_mode = AcquisitionType.FINITE,
                                            samps_per_chan = n_samples)
            
            ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{self.device}/ao/StartTrigger"
            )

            writer = AnalogSingleChannelWriter(ao_task.out_stream)
            writer.write_many_sample(output_waveform)

            reader = AnalogMultiChannelReader(ai_task.in_stream)

            ai_task.start()
            ao_task.start()
            reader.read_many_sample(input_buffer, number_of_samples_per_channel=n_samples, timeout=duration + 5)

            if not silence:
                print("Synchronized acquisition completed.")

        return input_buffer, output_waveform, time_array
    
    def acquire_multi_channel_with_timeout_or_lockfile(self,
                                                    channels,
                                                    rate,
                                                    buffer_size,
                                                    max_duration,
                                                    lock_filename = "lock",
                                                    use_temp_lock = True,
                                                    silence = False) -> np.ndarray:
        """
        Acquires data from multiple analog input channels.

        - If use_temp_lock=True: Waits for lock file to appear before starting. Stops when it's removed or max_duration is exceeded.
        - If use_temp_lock=False: Starts immediately and stops after max_duration.

        Parameters:
            channels (List[str]): List of AI channels (e.g., ['ai0', 'ai1']).
            rate (int): Sampling rate in samples per second.
            buffer_size (int): Number of samples to read per iteration.
            max_duration (float): Maximum acquisition duration in seconds.
            lock_filename (str): Name of the lock file to check in the system temp directory.
            use_temp_lock (bool): Whether to use the lock file mechanism.
            silence (bool): If True, suppress console output.

        Returns:
            np.ndarray: Acquired data as (channels x samples)
        """
        lock_path = f"temp/{lock_filename}"

        def lock_exists():
            return os.path.exists(lock_path)

        if use_temp_lock:
            if not silence:
                print(f"Waiting for lock file to appear: {lock_path}")
            while not lock_exists():
                time.sleep(0.2)
            if not silence:
                print("Lock file detected. Starting acquisition...")

        start_time = time.time()
        data_chunks = []

        with nidaqmx.Task() as task:
            for ch in channels:
                task.ai_channels.add_ai_voltage_chan(f"{self.device}/{ch}",
                                                    terminal_config=TerminalConfiguration.DIFF)

            task.timing.cfg_samp_clk_timing(rate=rate,
                                            sample_mode=AcquisitionType.CONTINUOUS,
                                            samps_per_chan=buffer_size)

            reader = AnalogMultiChannelReader(task.in_stream)
            buffer = np.zeros((len(channels), buffer_size))

            task.start()

            try:
                while (time.time() - start_time) < max_duration:
                    if use_temp_lock and not lock_exists():
                        if not silence:
                            print("Lock file removed. Stopping acquisition.")
                        break
                    reader.read_many_sample(buffer, number_of_samples_per_channel=buffer_size, timeout=10.0)
                    data_chunks.append(buffer.copy())
            except Exception as e:
                if not silence:
                    print(f"Acquisition interrupted by exception: {e}")
            finally:
                if not silence:
                    print("Multi-channel acquisition stopped.")

        if data_chunks:
            return np.concatenate(data_chunks, axis=1)
        else:
            return np.empty((len(channels), 0))
