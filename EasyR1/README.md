### Software Requirements

- Python 3.9+
- transformers>=4.51.0
- flash-attn>=2.4.3
- vllm>=0.8.3

We provide a [Dockerfile](./Dockerfile) to easily build environments.

We recommend using the [pre-built docker image](https://hub.docker.com/r/hiyouga/verl) in EasyR1.

```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

If your environment does not support Docker, you can consider using **Apptainer**:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

### Installation

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

### Generate RL datasets

```bash
cd EasyR1/scripts

bash VRSBench2json.sh

bash RSVQA2json.sh

bash RSVG2json.sh

bash FIT-RS-VQA2json.sh

#use hebing.py to Merge the dataset
python hebing.py --src <path/to/datasets1> <path/to/datasets2>··· --dst your/dataset/root
```

### GRPO Training

Before running, first check that all dataset、model paths inside the `.sh` files are correct.

```bash
#for Grounding
bash examples/run/OursModelv4_ground.sh
#for Choice
bash examples/run/OursModelv4_choice.sh
#for VQA
bash examples/run/OursModelv4_vqa.sh
```
