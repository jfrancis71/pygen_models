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

