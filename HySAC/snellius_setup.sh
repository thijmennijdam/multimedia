module purge
module load 2024
module load Anaconda3/2024.06-1

conda create -n safe-clip python==3.10
source activate safe-clip
pip install -r requirements.txt
pip install -e .
pip install umap-learn
python test.py 


module purge
module load 2024
module load Anaconda3/2024.06-1
source activate safe-clip
