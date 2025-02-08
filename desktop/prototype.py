import tkinter as tk
import threading
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from transformers import AutoModelForCausalLM, WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer  # Tokenizer still from transformers
import torch
import edge_tts
import asyncio
import time
from pathlib import Path
import psutil
import logging  # Import logging module

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: RSS = {memory_info.rss / (1024 * 1024):.2f} MB, VMS = {memory_info.vms / (1024 * 1024):.2f} MB")


class VoiceRecorder:
    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.frames = []
        self.stream = None

    def callback(self, indata, frames, time, status):
        if status:
            print("Status:", status) # Keep basic status print for sounddevice
        # Use the first channel (assuming mono audio)
        self.frames.append(indata[:, 0].copy())

    def start_recording(self):
        self.frames = []
        try: # Add error handling for stream start
            self.stream = sd.InputStream(samplerate=self.samplerate,
                                            channels=self.channels,
                                            dtype='float32',
                                            callback=self.callback)
            self.stream.start()
        except Exception as e:
            logging.error(f"Error starting audio stream: {e}")
            self.stream = None # Ensure stream is None in case of error

    def stop_recording(self):
        if self.stream:
            try: # Add error handling for stream stop/close
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error stopping audio stream: {e}")
        if not self.frames:  # No frames recorded
            return None
        audio_data = np.concatenate(self.frames, axis=0)
        return audio_data


class ChatApp:
    def __init__(self, master):
        self.master = master
        master.title("中文聊天助手")

        # Conversation display area (read-only)
        self.text_display = tk.Text(master, height=20, width=60, state=tk.NORMAL, font=('Microsoft YaHei', 10))
        self.text_display.pack(padx=10, pady=10)
        # Configure tag for "thinking" message style
        self.text_display.tag_config('thinking', font=('Microsoft YaHei', 10, 'italic'), foreground="grey")

        # Entry field for text input
        self.entry = tk.Entry(master, width=50, font=('Microsoft YaHei', 10))
        self.entry.pack(padx=10, pady=(0, 10))

        # Send button for text
        self.send_button = tk.Button(master, text="发送", command=self.send_text, font=('Microsoft YaHei', 10))
        self.send_button.pack(padx=10, pady=(0, 10), side=tk.LEFT) # side=tk.LEFT to place buttons horizontally

        # Clear history button
        self.clear_button = tk.Button(master, text="清空聊天记录", command=self.clear_history, font=('Microsoft YaHei', 10))
        self.clear_button.pack(padx=10, pady=(0, 10), side=tk.LEFT) # side=tk.LEFT to place buttons horizontally


        # "Hold to Talk" button for voice input
        self.talk_button = tk.Button(master, text="按住说话", width=20, font=('Microsoft YaHei', 10))
        self.talk_button.pack(padx=10, pady=(0, 10), side=tk.LEFT) # side=tk.LEFT to place buttons horizontally
        self.talk_button.bind("<ButtonPress-1>", self.on_hold_start)
        self.talk_button.bind("<ButtonRelease-1>", self.on_hold_end)

        # Voice selection for text-to-speech (TTS)
        self.voice_var = tk.StringVar(value="zh-CN-XiaoyiNeural")
        voice_frame = tk.Frame(master)
        voice_frame.pack(padx=10, pady=(0, 10))
        tk.Label(voice_frame, text="选择声音：", font=('Microsoft YaHei', 10)).pack(side=tk.LEFT)
        voices = ["zh-CN-XiaoyiNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural"]
        voice_menu = tk.OptionMenu(voice_frame, self.voice_var, *voices)
        voice_menu.config(font=('Microsoft YaHei', 10))
        voice_menu.pack(side=tk.LEFT)

        # Initialize components
        self.voice_recorder = VoiceRecorder()

        # Create temporary directory for audio files if it doesn't exist
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)

        print("加载 Whisper 模型前:")
        print_memory_usage()

        # Initialize Whisper for speech recognition
        whisper_model = "openai/whisper-tiny"
        try: # Error handling for Whisper model loading
            self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                whisper_model,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            tk.messagebox.showerror("模型加载错误", f"加载 Whisper 模型失败: {e}") # Show error in UI
            logging.error(f"Whisper model loading error: {e}") # Log the error
            master.destroy() # Close the app if model loading fails is critical
            return  # Exit init to prevent further errors

        print("加载 Whisper 模型后:")
        print_memory_usage()


        # Initialize the DeepSeek model
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        try: # Error handling for DeepSeek model loading
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            tk.messagebox.showerror("模型加载错误", f"加载 DeepSeek 模型失败: {e}") # Show error in UI
            logging.error(f"DeepSeek model loading error: {e}") # Log the error
            master.destroy() # Close if DeepSeek loading fails
            return # Exit init

        print("开始语音识别前:")
        print_memory_usage()

        # Conversation history
        self.conversation_history = []
        self.max_history_turns = 5

    def on_hold_start(self, event):
        self.update_display("正在录音...\n")
        self.voice_recorder.start_recording()

    def on_hold_end(self, event):
        if self.voice_recorder.stream is None: # Check if stream started successfully
            self.update_display("录音启动失败，请检查麦克风设置。\n") # Inform user if recording failed to start
            return
        threading.Thread(target=self.process_voice_input, daemon=True).start()

    def process_voice_input(self):
        audio_np = self.voice_recorder.stop_recording()

        if audio_np is None:
            self.update_display("未检测到语音输入，请稍后重试。\n")
            return

        try:
            print("audio_np shape (before type conversion):", audio_np.shape)
            print("audio_np dtype (before type conversion):", audio_np.dtype)

            # Ensure audio is normalized float32
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32) / 32768.0

            print("audio_np shape (after type conversion):", audio_np.shape)
            print("audio_np dtype (after type conversion):", audio_np.dtype)
            print("Shape of audio_np before whisper_processor:", audio_np.shape)

            # Process audio with Whisper
            input_features = self.whisper_processor(
                audio_np,
                sampling_rate=self.voice_recorder.samplerate,
                return_tensors="pt",
                padding=True
            ).input_features

            print("Shape of input_features after whisper_processor:", input_features.shape)
            print("Data type of input_features after whisper_processor:", input_features.dtype)

            input_features = input_features.to(torch.float16).to(self.whisper_model.device)

            predicted_ids = self.whisper_model.generate(
                input_features,
                language="zh",
                task="transcribe"
            )

            recognized_text = self.whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            print("语音识别后:")
            print_memory_usage()

        except Exception as e: # More specific error handling
            recognized_text = ""
            logging.exception("Speech recognition processing error:") # Log full error traceback
            self.update_display(f"语音识别错误，请重试。(Error: {e})\n") # Inform user of error

        if recognized_text:
            self.update_display("用户（语音）: " + recognized_text + "\n")
            self.process_user_input(recognized_text)
        else:
            self.update_display("无法识别语音。\n")

    def send_text(self):
        text = self.entry.get().strip()
        if text:
            self.update_display("用户: " + text + "\n")
            self.entry.delete(0, tk.END)
            self.process_user_input(text)

    def process_user_input(self, user_text):
        threading.Thread(target=self.get_llm_response, args=(user_text,), daemon=True).start()

    def get_llm_response(self, user_text):
        # Append user input to conversation history
        self.conversation_history.append(f"用户: {user_text}")
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]

        # Format the prompt
        prompt = "\n".join(self.conversation_history) + "\n助手:"

        # Display "Thinking..." indicator
        self.update_display("助手正在思考...\n", animated=False, thinking_indicator=True)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generation config
        generation_config = {
            'max_new_tokens': 1024,
            'temperature': 0.7,
            'top_p': 0.9,
            'no_repeat_ngram_size': 3,
            'pad_token_id': self.tokenizer.eos_token_id,
        }

        try: # Error handling for LLM response generation
            with torch.no_grad():
                response_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **generation_config
                )

            # Decode and store response
            response_text = self.tokenizer.decode(
                response_ids[0],
                skip_special_tokens=True
            )

            print("模型返回的原始响应:")
            print(response_text)

            # Append the assistant's response to conversation history
            self.conversation_history.append(f"助手: {response_text}")

            # Update display and speak
            self.update_display("助手: " + response_text + "\n", animated=True, thinking_indicator=False)
            if response_text.find('</think>'):
                self.speak_text(response_text.split('</think>')[-1].strip())
            else:
                self.speak_text(response_text)

        except Exception as e: # Handle LLM generation errors
            response_text = "抱歉，助手在思考时遇到了问题，请稍后再试。" # User-friendly error message
            logging.exception("LLM response generation error:") # Log full error
            self.update_display("助手: " + response_text + "\n", animated=True, thinking_indicator=False) # Display error to user
            self.speak_text(response_text) # Optionally speak the error message


    def update_display(self, text, animated=True, thinking_indicator=False):
        if thinking_indicator:
            self.text_display.insert(tk.END, "助手正在思考...\n", 'thinking')
            self.text_display.see(tk.END)
        elif animated:
            self.text_display.insert(tk.END, "\n")
            self.animate_text(text)
        else:
            self.text_display.insert(tk.END, text + "\n")
            self.text_display.see(tk.END)

    def animate_text(self, text, index=0):
        if index < len(text):
            self.text_display.insert(tk.END, text[index])
            self.text_display.see(tk.END)  # Keep scrolled to bottom
            self.master.after(20, self.animate_text, text, index + 1)  # Insert one char every 20 ms

    def speak_text(self, text):
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        temp_file = self.temp_dir / f"speech_{int(time.time())}.mp3"

        async def _async_speak():
            try: # Error handling for TTS and audio playback
                communicate = edge_tts.Communicate(text, self.voice_var.get())
                await communicate.save(str(temp_file))
                import soundfile as sf
                data, samplerate = sf.read(str(temp_file))
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                logging.error(f"TTS or audio playback error: {e}")
                tk.messagebox.showerror("语音合成/播放错误", f"文本转语音或播放失败: {e}") # Show error in UI
            finally: # Ensure temp file cleanup even if speaking fails
                if temp_file.exists():
                    temp_file.unlink()

        asyncio.run(_async_speak())

    def clear_history(self):
        """Clears the conversation history and the text display."""
        self.conversation_history = []
        self.text_display.config(state=tk.NORMAL) # Enable text widget to modify it
        self.text_display.delete("1.0", tk.END) # Delete all text from start to end
        self.text_display.config(state=tk.NORMAL) # Disable text widget again


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
