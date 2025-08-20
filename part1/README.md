
pip install python3  #???

python3 -m venv myenv
source myenv/bin/activate

pip install requirements.txt

# 1. Install Kaggle CLI (if not already)
pip install kaggle
# Ensure ~/.kaggle/kaggle.json has your API credentials

# 2. Download CelebA dataset (approx. 200K images)
kaggle datasets download -d jessicali9530/celeba-dataset -p data/real --unzip

# 3. Download DFDC samples (requires competition entry)
kaggle competitions download -c deepfake-detection-challenge -p data/ai --unzip
