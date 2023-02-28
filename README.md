# PSL_GAN
This code is for generating a multisite stochastic weather generator using GAN

srun -n 1 --gres=gpu:1 -p gpu --container-mounts="/home/psl-gan/:/psl-gan" --container-workdir=/psl-gan/code --container-image registry.gitlab.com/asubramaniam/containers/psl-gan:latest -t 00:30:00 --pty bash -i -l

Running curiosity cluster: https://axis-raplabhackathon.axisportal.io/apps

IMAGE=registry.gitlab.com/cponder/containers/ubuntu-pgi-openmpi/selene:latest


srun --container-image $IMAGE --container-mounts $PWD --container-workdir $PWD -t 00:30:00 --pty bash -i -l

If your local cluster doesn’t have the latest Nsight version- here’s where you can download them:

Nsight Systems profiler (version 2023.1) for Windows or Linux: Nsight Systems 2023.1

Nsight Compute latest is 2022.4.1, which can be found here: https://developer.nvidia.com/gameworksdownload#?dn=nsight-compute-2022-4-1

Profiling guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model


NVTX: https://docs.nvidia.com/nvtx/overview/index.html

NVIDIA Nsight Systems user guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html

Apptainer: https://apptainer.org/docs/user/1.0/bind_paths_and_mounts.html

apptainer run /usr/local/containers/lolcow.sif

Videos: 

Nsight AI Tutorial | NOAA/NCAR Hackathon | NVIDIA https://www.youtube.com/watch?v=u8FcM1j5u_0

Nsight Compute + Nsight System Q&A + Demo | NOAA/NCAR Hackathon NVIDIA: https://www.youtube.com/watch?v=jBc4qNzXW-0

Tensorboard Tutorial NOAA/NCAR Hackathon NVIDIA: https://www.youtube.com/watch?v=9dMWBAg3Yd8
