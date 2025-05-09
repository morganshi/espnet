# from espnet_model_zoo.downloader import ModelDownloader
# d = ModelDownloader("./downloads")
# d.download_and_unpack("kan-bayashi/ljspeech_conformer_fastspeech2")
# Download vocoder
from parallel_wavegan.utils import download_pretrained_model
download_pretrained_model("ljspeech_hifigan.v1", "downloads")