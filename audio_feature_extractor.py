import librosa
import numpy as np
import torch.cuda
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class AudioFeatureExtractor:
    feature_extractor = None
    audio_model = None
    device = None

    def __init__(self, audio_model_name, device="cpu"):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name, device=device)
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name).to(device)
        self.device = device

    def _get_audio_streams(self, file_path, original_sample_rate, chunk_size_seconds=60):
        frame_length = int(original_sample_rate)
        hop_length = int(original_sample_rate)  # no striding

        audio_streams = librosa.stream(file_path,
                                       block_length=chunk_size_seconds,
                                       frame_length=frame_length,
                                       hop_length=hop_length)

        return audio_streams

    def extract_long_audio_embedding(self, file_path) -> np.array:
        original_sample_rate = librosa.get_samplerate(file_path)
        resample_sample_rate = self.feature_extractor.sampling_rate

        chunk_embeddings = []  # contains the features for each 60 seconds long audio chunk
        splitted_audio_chunks = self._get_audio_streams(file_path, original_sample_rate, chunk_size_seconds=60)
        for audio_chunk in splitted_audio_chunks:
            # resample the chunk and convert it to a pytorch tensor
            audio_chunk = librosa.resample(audio_chunk, orig_sr=original_sample_rate, target_sr=resample_sample_rate)
            audio_chunk = torch.tensor(audio_chunk).to(self.device)

            # extract the features to feed the audio model
            extractor_data = self.feature_extractor(audio_chunk, sampling_rate=self.feature_extractor.sampling_rate,
                                                    padding=True, return_tensors="pt")

            with torch.no_grad():
                try:
                    # get audio model features
                    model_output = self.audio_model(extractor_data.input_values.to(self.device))

                    # the audio model returns a 512 elements array for each tiny sample in the audio
                    # we are interested in having a single feature for the whole audio file
                    # the features for each tiny sample are combined by a simple mean

                    chunk_features = torch.mean(model_output.extract_features, axis=1)
                    chunk_features = np.array(chunk_features.cpu())  # convert the 512 elements array to numpy
                    chunk_embeddings.append(chunk_features)
                except:
                    print("Chunk processing error")

        # combines the features of each 60 seconds long chunk by averaging the embeddings
        file_embedding = np.mean(chunk_embeddings, axis=0)
        return file_embedding
