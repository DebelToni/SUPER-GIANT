git pull
apt update
apt install -y vim
pythom -m pip install --upgrade pip
pip install -r ../requirements.txt
pip install --upgrade jax[cuda12] jaxlib flax
pip install flash-attn-jax
