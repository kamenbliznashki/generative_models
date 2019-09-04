



Collection of generative methods in pytorch.

# Implemented models
* [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) / [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
* [Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](https://arxiv.org/abs/1603.08575v3)
* [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623.pdf)
* [Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298)
* [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) (DCGAN)
* [Auto-encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

The models are implemented for MNIST data; other datasets are a todo.

## Dependencies
* python 3.6
* pytorch 0.4.1+
* numpy
* matplotlib
* tensorboardx
* tqdm

###### Some of the models further require
* observations
* imageio


## VQ-VAE2

Implementation of Generating Diverse High-Fidelity Images with VQ-VAE-2 (https://arxiv.org/abs/1906.00446) based on Vector Quantised VAE per Neural Discrete Representation Learning (https://arxiv.org/abs/1711.00937) with PixelCNN prior on the level 1 discrete latent variables per Conditional Image Generation with PixelCNN Decoders (https://arxiv.org/abs/1606.05328) and PixelSNAIL prior on the level 2 discrete latent variables per PixelSNAIL: An Improved Autoregressive Generative Model (https://arxiv.org/abs/1712.09763).

#### Results

Model reconstructions on [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) Chest X-Ray Dataset -- CheXpert is a large public dataset for chest radiograph interpretation, consisting of 224,316 chest radiographs of 65,240 patients at 320x320 pixels for the small version of the dataset. Reconstructions and samples below are for 128x128 images using a codebook of size 8 (3 bits) which for single channel 8-bit gray scale images from CheXpert results to similar compression ratios listed in paper for 8-bit RGB images.

| Original | Bottom reconstruction <br> (using top encoding) | Top reconstruction <br> (using zeroed out bottom encoding) |
| --- | --- | --- |
| ![chexpertoriginal](images/vqvae2/128x128_bits3_eval_reconstruction_step_87300_original.png) | ![vqvae2bottomrecon](images/vqvae2/128x128_bits3_eval_reconstruction_step_87300_bottom.png) | ![vqvae2toprecon](images/vqvae2/128x128_bits3_eval_reconstruction_step_87300_top.png) |


##### Model samples from priors
Both top and bottom prior models are pretty heavy to train; the samples below were trained only for 84k and 140k steps for top and bottom priors, respectively, using smaller model sizes than what was reported in the paper. The samples are class conditional along the rows for classes (atelectasis, cardiomegaly, consolidation, edema, pleural effusion, no finding) -- much more to be desired / improved with larger models and higher computational budget.

Model parameters:
* bottom prior: n_channels 128, n_res_layers 20, n_cond_stack_layers 10, drop_rate 0.1, batch size 16, lr 5e-5
* top prior: n_channels 128, n_res_layers 5, n_out_stack_layers 10, drop_rate 0.1, batch size 128, lr 5e-5; (attention params: layers 4, heads 8, dq 16, dk 16, dv 128)

![vqvae2_sampes](images/vqvae2/generation_sample_step_52440_top_b128_c128_outstack10_bottom_b16_c128_nres20_condstack10.png)

#### Usage

To train and evaluate/reconstruct from the VAE model with hyperparameters of the paper:
```
python vqvae.py --train
                --n_embeddings [size of the latent space]
                --n_epochs     [number of epochs to train]
                --ema          [flag to use exponential moving average training for the embeddings]
                --cuda         [cuda device to run on]

python vqvae.py --evaluate
                --restore_dir [path to model directory with config.json and saved checkpoint]
                --n_samples   [number of examples from the validation set to reconstruct]
                --cuda [cuda device to run on]
```

To train the top and bottom priors on the latent codes using 4 GPUs and Pytorch DistributedDataParallels:
* the latent codes are extracted for the full dataset and saved as a pytorch dataset object, which is then loaded into memory for training
* hyperparameters not shown as options below are at the defaults given by the paper (e.g. kernel size, attention parameters)

```
python -m torch.distributed.launch --nproc_per_node 4 --use_env \
  vqvae_prior.py --vqvae_dir              [path to vae model used for encoding the dataset and decoding samples]
                 --train
                 --distributed            [flag to use DistributedDataParallels]
                 --n_epochs 20
                 --batch_size 128
                 --lr 0.00005
                 --which_prior top        [which prior to train]
                 --n_cond_classes 5       [number of classes to condition on]
                 --n_channels 128         [convolutional channels throughout the architecture]
                 --n_res_layers 5         [number of residual layers]
                 --n_out_stack_layers 10  [output convolutional stack (used only by top prior)]
                 --n_cond_stack_layers 0  [input conditional stack (used only by bottom prior)]
                 --drop_rate 0.1          [dropout rate used in the residual layers]

python -m torch.distributed.launch --nproc_per_node 4 --use_env \
  vqvae_prior.py --vqvae_dir [path_to_vae_directory]
                 --train
                 --distributed
                 --n_epochs 20
                 --batch_size 16
                 --lr 0.00005
                 --which_prior bottom
                 --n_cond_classes 5
                 --n_channels 128
                 --n_res_layers 20
                 --n_out_stack_layers 0
                 --n_cond_stack_layers 10
                 --drop_rate 0.1
```

To generate data from a trained model and priors:
```

python -m torch.distributed.launch --nproc_per_node 4 --use_env \
  vqvae_prior.py --vqvae_dir    [path_to_vae_directory]
                 --restore_dir  [path_to_bottom_prior_directory, path_to_top_prior_directory]
                 --generate
                 --distributed
                 --n_samples    [number of samples to generate (per gpu)]
```

Useful resources
* Official tensorflow implementation of the VQ layer and VAE model in Sonnet (https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb)

## AIR

Reimplementation of the Attend, Infer, Repeat (AIR) architecture.
https://arxiv.org/abs/1603.08575v3

#### Results
Model reconstructed data (top row is sample of original images, bottom row is AIR reconstruction; red attention window corresponds to first time step, green to second):

![air_recon](images/air/image_recons_270.png)

EBLO and object count accuracy after 300 epochs of training using RMSprop with the default hyperparameters discussed in the paper and linear annealing of the z_pres probability. Variance coming from the discrete z_pres is alleviated using NVIL ([Mnih & Gregor](https://arxiv.org/abs/1402.0030)) but can still be seen in the count accuracy in the first 50k training iterations.


| Variational bound | Count accuracy |
| --- | --- |
| ![air_elbo](images/air/air_elbo.png) | ![air_count](images/air/air_count.png)

#### Usage
To train a model with hyperparameters of the paper:
```
python air.py -- train \
              -- cuda=[# of cuda device to run on]
```

To evaluate model ELBO:
```
python air.py -- evaluate \
              -- restore_file=[path to .pt checkpoint]
              -- cuda=[# of cuda device to run on]
```

To generate data from a trained model:
```
python air.py -- generate \
              -- restore_file=[path to .pt checkpoint]
```

Useful resources
* tensorflow implementation https://github.com/akosiorek/attend_infer_repeat and by the same author Sequential AIR (a state-space model on top of AIR) (https://github.com/akosiorek/sqair/)
* pyro implmentation and walk through http://pyro.ai/examples/air.html

## DRAW
Reimplementation of the Deep Recurrent Attentive Writer (DRAW) network architecture. https://arxiv.org/abs/1502.04623

#### Results
Model generated data:

![draw](images/draw/generated_32_time_steps.gif)

Results were achieved training at the parameters presented in the paper (except at 32 time steps) for 50 epochs.

Visualizing the specific filterbank functions for read and write attention (cf Figure 3 & 4 in paper):

| Extracted patches with grid filters | Applying transposed filters to reconstruct extracted image patch |
| --- | --- |
| ![drawread](images/draw/draw_fig_3.png) | ![drawwrite](images/draw/draw_fig_4.png)

#### Usage
To train a model with read and write attention (window sizes 2 and 5):
```
python draw.py -- train \
               -- use_read_attn \
               -- read_size=2 \
               -- use_write_attn \
               -- write_size=5 \
               -- [add'l options: e.g. n_epoch, z_size, lstm_size] \
               -- cuda=[# of cuda device to run on]
```

To evaluate model ELBO:
```
python draw.py -- evaluate \
               -- restore_file=[path to .pt checkpoint]
               -- [model parameters: read_size, write_size, lstm_size, z_size]
```

To generate data from a trained model:
```
python draw.py -- generate \
               -- restore_file=[path to .pt checkpoint]
               -- [model parameters: read_size, write_size, lstm_size, z_size]
```

#### Useful resources
* https://github.com/jbornschein/draw
* https://github.com/ericjang/draw


## Semi-supervised Learning with Deep Generative Models
https://arxiv.org/abs/1406.5298

Reimplementation of M2 model on MNIST.

#### Results
Visualization of handwriting styles learned by the model (cf Figure 1 in paper). Column 1 shows an image column from the test data followed by model generated data. Columns 2 and 3 show model generated styles for a fixed label and a linear variation of each component of a 2-d latent variable.

| MNIST analogies | Varying 2-d latent z (z1) on number 2 | Varying 2-d latent z (z2) on number 4 |
| --- | --- | --- |
| ![analogies](images/ssvae/analogies_sample.png) | ![c1](images/ssvae/latent_var_grid_sample_c1_y2.png) | ![c2](images/ssvae/latent_var_grid_sample_c2_y4.png)

#### Usage
To train a model:
```
python ssvae.py -- train \
                -- n_labeled=[100 | 300 | 1000 | 3000] \
                -- [add'l options: e.g. n_epochs, z_dim, hidden_size] \
                -- cuda=[# of cuda device to run on]
```

To evaluate model accuracy:
```
python ssvae.py -- evaluate \
                -- restore_file=[path to .pt checkpoint]
```

To generate data from a trained model:
```
python ssvae.py -- generate \
                -- restore_file=[path to .pt checkpoint]
```

#### Useful resource
* https://github.com/dpkingma/nips14-ssl


## InfoGAN

Reimplementation of InfoGan. https://arxiv.org/abs/1606.03657
This follows closely the Tensorflow implementation by [Depth First Learning](http://www.depthfirstlearning.com/2018/InfoGAN) using tf.distribution, which make the model quite intuitive.

#### Results

Visualizing model-generated data varying each component of a 2-d continuous latent variable:

| Varying 2-d latent z (z1)| Varying 2-d latent z (z2) |
| --- | --- |
| ![c1](images/infogan/latent_var_grid_sample_c1.png) | ![c2](images/infogan/latent_var_grid_sample_c2.png)

#### Usage
To train a model with read and write attention (window sizes 2 and 5):
```
python infogan.py -- n_epochs=[# epochs] \
               -- cuda=[# of cuda device to run on]
               -- [add'l options: e.g. noise_dim, cat_dim, cont_dim] \
```

To evaluate model and visualize latents:
```
python infogan.py -- evaluate_on_grid \
               -- restore_file=[path to .pt checkpoint]
```

#### Useful resources
* http://www.depthfirstlearning.com/2018/InfoGAN


## DCGAN

Reimplementation of DCGAN. https://arxiv.org/abs/1511.06434

#### Results
Model generated data:

![dcgan](images/dcgan/latent_var_grid_sample_c1.png)

#### Usage
To train a model with read and write attention (window sizes 2 and 5):
```
python infogan.py -- n_epochs=[# epochs] \
               -- cuda=[# of cuda device to run on]
               -- [add'l options: e.g. noise_dim, cat_dim, cont_dim] \
```

To evaluate model and visualize latents:
```
python infogan.py -- evaluate_on_grid \
               -- restore_file=[path to .pt checkpoint]
```

#### Useful resources
* pytorch code examples https://github.com/pytorch/examples/


## Auto-encoding Variational Bayes
Reimplementation of https://arxiv.org/abs/1312.6114

#### Results

Visualizing reconstruction (after training for 25 epochs):

| Real samples (left) and model reconstruction (right) |
| --- |
| ![vae_recon](images/basic_vae/reconstruction_at_epoch_24.png) |

Visualizing model-generated data and TSNE embedding in latent space:

| Model-generated data using Normal(0,1) prior | TSNE embedding in latent space |
| --- | --- |
| ![vae_sample](images/basic_vae/sample_at_epoch_24.png) | ![vae_tsne](images/basic_vae/tsne_embedding.png) |


#### Usage
To train and evaluate a model on MNIST:
```
python basic_vae.py -- n_epochs=[# epochs] mnist
```

#### Useful resources
* Implementation in Pyro and quick tutorial http://pyro.ai/examples/vae.html
