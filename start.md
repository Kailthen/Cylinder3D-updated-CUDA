### env
conda create -n pvkd python=3.8

conda activate pvkd
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

conda install pytorch-scatter -c pyg

### cumm
git clone https://github.com/FindDefinition/cumm
git checkout v0.28

export CUDA_HOME=/usr/local/cuda-11.8/
export PATH=${CUDA_HOME}/bin:${PATH}
export CUMM_CUDA_VERSION="11.8"
export CUMM_DISABLE_JIT="1"

pip install -e .

### spconv

git checkout v2.1.22
export SPCONV_DISABLE_JIT = "1"
pip install -e .

### prepare dataset
https://github.com/traveller59/second.pytorch

cd /media/work/2t/fangyuan/road-understanding/lidar/semantic/second.pytorch
export PYTHONPATH=.:${PYTHONPATH}
python second/create_data.py nuscenes_data_prep \
  --root_path=/media/work/2t/nuscenes_mini-pvkd \
  --version="v1.0-mini" \
  --max_sweeps=10 \
  --dataset_name="NuScenesDataset"


# demo
python demo_folder.py --demo-folder /media/work/2t/slam_results/fast-lio-slam/Scans --save-folder /media/work/2t/slam_results/fast-lio-slam/Scans-seg

# vis o3d
