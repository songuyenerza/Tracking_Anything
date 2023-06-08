## Install

python = 3.9.16
<!-- env : environment.yml -->
```bash
#rm-rf GroundingDINO
conda create -n moving python=3.9.16
conda activate moving
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=10.2 -c pytorch
pip3 install -q -e .
cd ..
pip3 install -r requirements.txt

```
Clone weight: 
```bash
cd GroundingDINO
mkdir CP
cd CP
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```

DownLoad Video demo:

```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qfKYcSfpXgXg1dvzhP1VRo1__hY6bU2z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qfKYcSfpXgXg1dvzhP1VRo1__hY6bU2z" -O video1_1.mp4 && rm -rf /tmp/cookies.txt    #video1

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_CgK3MQ-6IF1HTMxLX-fJ_bnxnaaQFwc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_CgK3MQ-6IF1HTMxLX-fJ_bnxnaaQFwc" -O video1_2.mp4 && rm -rf /tmp/cookies.txt    #video2

```

## How To Run:
Run Tracking:
```bash
bash ./scripts/run.sh
```
            