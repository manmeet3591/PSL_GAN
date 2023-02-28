# PSL_GAN
This code is for generating a multisite stochastic weather generator using GAN

srun -n 1 --gres=gpu:1 -p gpu --container-mounts="/home/psl-gan/:/psl-gan" --container-workdir=/psl-gan/code --container-image registry.gitlab.com/asubramaniam/containers/psl-gan:latest -t 00:30:00 --pty bash -i -l

Running curiosity cluster: https://axis-raplabhackathon.axisportal.io/apps
