## Quickstart

```bash
git clone https://github.com/mikky1sCp/zkash-10k.git
cd zkash-10k

# (опционально) venv
python -m venv .venv
source .venv/Scripts/activate     # Windows / Git Bash
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
pip install -e .

bash scripts/train.sh
bash scripts/eval.sh
pytest -q
```

### Expected output

```
[Zkash-10K] trainable params: 10000
...
val acc: 1.0000
saved checkpoint → checkpoints/zkash10k.pt
params : 10000
acc    : 1.0000
```