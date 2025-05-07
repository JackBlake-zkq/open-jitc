# Open Just in Time Checkpointing (JITC)

This repo is an open source implementation of the recent Microsoft Research paper "Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures". The main idea is to **checkpoint after a failure occurs**, exploiting the fact that the data parallel ranks have the same local weights. This can efficiently deal with recoverable errors where the GPU is till usable, but requires a restart or the like, which were about 70% of errors found during the training of OPT-175B.


## User Level Mode

We successfully replciate the "User Level" version on top of PyTorch Distributed Data Parallel (DDP). See `/UserLevel` for how to run this. This mode requires using the modified training logic provided, along with preloading a binary using `LD_PRELOAD`.

We have tested our implementation on Google Cloud with:



Hardware: L4 GPU(s)

OS Image: Deep Learning on Linux

OS Image Version: Deep Learning VM with CUDA 12.2 M126


## Transparent Mode

We are currently working on the "Transparent Mode" of JITC. This mode is much less tightly coupled to a particular training framework. It acts like a proxy that handles errors and restores the GPU without the training framework ever knowing the error occurred.

We'd like to note that, unfortunately, we have concluded that "Transparent Mode" as the JITC paper implements it, does actually require some changes to user code to be practical. We will soon add a report to this repo with details about this.

We plan to compare the performance of this approach to what we see as the next best way to deal with a recoverable errors, which we believe is to use a "Resilient Training" framework, which reconfigure the parallelism strategy when a failure occurs. This approach deals with unrecoverable errors, where the GPU is lost forever, better than any other approach and without backup hardware. The most recent iterations of this approach are ReCycle and Oobleck (most recent open source approach). We'd like to compare transparent mode to re-adding a GPU into Oobleck after it is recovered. We suspect JITC is much quicker.

We'd also like to test integration and overhead with common training frameworks such as [DeepSpeed](https://github.com/deepspeedai/DeepSpeed).


# Citations
@inproceedings{10.1145/3627703.3650085,
author = {Gupta, Tanmaey and Krishnan, Sanjeev and Kumar, Rituraj and Vijeev, Abhishek and Gulavani, Bhargav and Kwatra, Nipun and Ramjee, Ramachandran and Sivathanu, Muthian},
title = {Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures},
year = {2024},
isbn = {9798400704376},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627703.3650085},
doi = {10.1145/3627703.3650085},
abstract = {Deep Learning training jobs process large amounts of training data using many GPU devices, often running for weeks or months. When hardware or software failures happen, these jobs need to restart, losing the memory state for the Deep Neural Network (DNN) model trained so far, unless checkpointing mechanisms are used to save training state periodically. However, for large models, periodic checkpointing incurs significant steady state overhead, and during recovery, a large number of GPUs need to redo work since the last checkpoint. This is especially problematic when failures are frequent for large DNN (such as Large Language Model) training jobs using many GPUs. In this paper, we present a novel approach of just-in-time checkpointing when failures happen, which enables recovery from failures with just a single minibatch iteration of work replayed by all GPUs. This reduces the cost of error recovery from several minutes to a few seconds per GPU, with nearly zero steady state overhead. This also avoids the guesswork of choosing a checkpointing frequency since failure rates usually have high variance. We discuss how just-in-time checkpointing can be enabled in training code, as well as design of key mechanisms for transparent just-in-time checkpointing without user code change. We analyze the wasted GPU work of just-in-time checkpointing and show that it is less than periodic checkpointing for large numbers of GPUs. We present results from our implementation in modern AI cluster infrastructure.},
booktitle = {Proceedings of the Nineteenth European Conference on Computer Systems},
pages = {1110–1125},
numpages = {16},
keywords = {Large Scale DNN Training Reliability, Reliable Distributed Systems, Systems for Machine Learning},
location = {Athens, Greece},
series = {EuroSys '24}
}

@inproceedings{Gandhi_2024, series={SOSP ’24},
   title={ReCycle: Resilient Training of Large DNNs using Pipeline Adaptation},
   url={http://dx.doi.org/10.1145/3694715.3695960},
   DOI={10.1145/3694715.3695960},
   booktitle={Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles},
   publisher={ACM},
   author={Gandhi, Swapnil and Zhao, Mark and Skiadopoulos, Athinagoras and Kozyrakis, Christos},
   year={2024},
   month=nov, pages={211–228},
   collection={SOSP ’24} }


@inproceedings{10.1145/3600006.3613152,
author = {Jang, Insu and Yang, Zhenning and Zhang, Zhen and Jin, Xin and Chowdhury, Mosharaf},
title = {Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates},
year = {2023},
isbn = {9798400702297},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3600006.3613152},
doi = {10.1145/3600006.3613152},
abstract = {Oobleck enables resilient distributed training of large DNN models with guaranteed fault tolerance. It takes a planning-execution co-design approach, where it first generates a set of heterogeneous pipeline templates and instantiates at least f + 1 logically equivalent pipeline replicas to tolerate any f simultaneous failures. During execution, it relies on already-replicated model states across the replicas to provide fast recovery. Oobleck provably guarantees that some combination of the initially created pipeline templates can be used to cover all available resources after f or fewer simultaneous failures, thereby avoiding resource idling at all times. Evaluation on large DNN models with billions of parameters shows that Oobleck provides consistently high throughput, and it outperforms state-of-the-art fault tolerance solutions like Bamboo and Varuna by up to 13.9\texttimes{}.},
booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles},
pages = {382–395},
numpages = {14},
keywords = {fault tolerant training, distributed training, hybrid parallelism, pipeline template},
location = {Koblenz, Germany},
series = {SOSP '23}
}
