import whisper
import tempfile
import os
from threading import Thread
import datetime
from utils.logger_manager import logger
import time


class WhisperTranscriber:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.transcript_file = output_path.replace(".mp4", ".txt")
        self.srt_file = output_path.replace(".mp4", ".srt")
        self.model = whisper.load_model("turbo")
        logger.info(f"Transcript will be saved to: {self.transcript_file}")

    def transcribe_file(self):
        if not os.path.exists(self.output_path):
            logger.error(f"File does not exist: {self.output_path}")
            return

        try:
            import subprocess

            # Extract WAV audio from MP4 using ffmpeg
            process = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    self.output_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            audio_data = process.stdout
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                wav_file.write(audio_data)
                wav_path = wav_file.name

            # Transcribe the extracted audio
            result = self.model.transcribe(wav_path)
            os.remove(wav_path)

            with open(self.transcript_file, "w", encoding="utf-8") as txt_out, open(
                self.srt_file, "w", encoding="utf-8"
            ) as srt_out:
                for i, segment in enumerate(result.get("segments", []), start=1):
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"]
                    # Write to .txt
                    txt_out.write(f"[{start:.2f} --> {end:.2f}] {text}\n")

                    # Format timestamps for SRT
                    def format_time(t):
                        hours = int(t // 3600)
                        minutes = int((t % 3600) // 60)
                        seconds = int(t % 60)
                        milliseconds = int((t - int(t)) * 1000)
                        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

                    srt_out.write(f"{i}\n")
                    srt_out.write(f"{format_time(start)} --> {format_time(end)}\n")
                    srt_out.write(f"{text.strip()}\n\n")

            logger.info(f"Transcription complete: {self.transcript_file}")

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")

    @classmethod
    def transcribe(cls, output_path: str):
        instance = cls(output_path)
        instance.transcribe_file()
