from cog import BasePredictor, Input, Path
from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        preload_models()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests "
            "such as playing tic tac toe.",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        audio_array = generate_audio(prompt, output_full=True)
        output = "/tmp/audio.wav"
        write_wav(output, SAMPLE_RATE, audio_array)

        return Path(output)
