ARG PYT_VER=23.01
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3

RUN pip install pandas matplotlib seaborn
