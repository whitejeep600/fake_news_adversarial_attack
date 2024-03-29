# Requires conda, poetry
# Installs torch with both CPU and GPU support, so help me god

conda env create -f environment.yaml
conda activate adversarial
poetry install

# Note that if rerunning poetry install, this command needs to be rerun
# afterwards too, otherwise there will be no GPU support
pip install torch==2.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113

mypy --install-types
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"