# Results

### PixelCNN (using defaults)

score is validation epoch (in nats).

|net type|log_prob|
|----|--------|
|simple pixelcnn|-127|
|pixelcnn_net num_resnet=1|-54|

PixelCNN with num_resnet=1

![PixelCNN](https://github.com/jfrancis71/pygen_models/blob/documentation/results/pixelcnn.png?raw=true)

### Conditional PixelCNN (using defaults)

This is a generative model over digit images conditioned on knowledge of the digit class.

score is validation epoch (in nats).

|net type|log_prob|
|----|--------|
|simple pixelcnn|-91|
|pixelcnn_net num_resnet=1|-53|

Conditional PixelCNN with num_resnet=1

![PixelCNN](https://github.com/jfrancis71/pygen_models/blob/documentation/results/conditional_generated_mnist.png?raw=true)

### Discrete VAE

This is a generative model over digit images.

score is validation epoch (in nats).

Trained with simple_pixelcnn with 128 Bernoulli variables. discrete_vae_beta was trained with beta=0.1

|net type|log_prob|kl_div|reconstruct_log_prob|generative|reconstruct|
|----|--------|-------|-----------|-------------|------------------|
|discrete_vae|-89|7.2|-82|![Discrete_VAE_generated](https://github.com/jfrancis71/pygen_models/blob/documentation/results/discrete_vae_generated.png?raw=true)|![Discrete_VAE_reconstruct](https://github.com/jfrancis71/pygen_models/blob/documentation/results/discrete_vae_reconstruct.png?raw=true)|
|discrete_vae_beta|-105|58|-46|![Discrete_VAE_beta_generated](https://github.com/jfrancis71/pygen_models/blob/documentation/results/discrete_vae_beta_generated.png?raw=true)|![Discrete_VAE_beta_reconstruct](https://github.com/jfrancis71/pygen_models/blob/documentation/results/discrete_vae_beta_reconstruct.png?raw=true)|

