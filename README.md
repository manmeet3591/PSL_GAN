# PSL_GAN
This code is for generating a multisite stochastic weather generator using GAN

srun -n 1 --gres=gpu:1 -p gpu --container-mounts="/home/psl-gan/:/psl-gan" --container-workdir=/psl-gan/code --container-image registry.gitlab.com/asubramaniam/containers/psl-gan:latest -t 00:30:00 --pty bash -i -l

Running curiosity cluster: https://axis-raplabhackathon.axisportal.io/apps

IMAGE=registry.gitlab.com/cponder/containers/ubuntu-pgi-openmpi/selene:latest


srun --container-image $IMAGE --container-mounts $PWD --container-workdir $PWD -t 00:30:00 --pty bash -i -l

If your local cluster doesn’t have the latest Nsight version- here’s where you can download them:

Nsight Systems profiler (version 2023.1) for Windows or Linux: Nsight Systems 2023.1

Nsight Compute latest is 2022.4.1, which can be found here: https://developer.nvidia.com/gameworksdownload#?dn=nsight-compute-2022-4-1
