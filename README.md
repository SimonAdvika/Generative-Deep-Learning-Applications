# AI-Based Handwritten Digit Generation Using GANs

## Generative Deep Learning Application

This project implements a **Generative Adversarial Network (GAN)** using **TensorFlow/Keras** to generate new handwritten digit images similar to the MNIST dataset.

The project demonstrates the basic concepts of Generative AI, adversarial learning, Generator and Discriminator networks, image preprocessing, model training, and generated-image evaluation.

---

## Project Objective

The main objectives of this project are:

* Understand the fundamentals of Generative AI.
* Understand the architecture of GANs.
* Load and preprocess the MNIST dataset.
* Build a Generator neural network.
* Build a Discriminator neural network.
* Train both networks using adversarial learning.
* Generate new handwritten digit images.
* Visualize generated images during training.
* Analyze Generator and Discriminator loss curves.

---

## Dataset

### MNIST Handwritten Digit Dataset

The MNIST dataset contains grayscale handwritten digit images from 0 to 9.

### Dataset characteristics

* Training images: 60,000
* Image size: 28 × 28 pixels
* Channels: 1 grayscale channel
* Classes: 10 digits
* Pixel range before preprocessing: 0–255

The dataset is loaded directly using TensorFlow/Keras:

```python
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
```

The labels are not required because GAN training does not use the digit labels.

---

## Data Preprocessing

The MNIST images are converted to `float32` and normalized from:

```text
0 to 255
```

to:

```text
-1 to 1
```

This matches the output range of the Generator's final `tanh` activation.

The images are also reshaped from:

```text
28 × 28
```

to:

```text
28 × 28 × 1
```

Example:

```python
x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
```

---

## GAN Architecture

The GAN contains two neural networks:

### 1. Generator

The Generator receives a random noise vector containing 100 values and produces a 28 × 28 grayscale image.

Architecture:

```text
100-dimensional Noise
        ↓
Dense
        ↓
Batch Normalization
        ↓
LeakyReLU
        ↓
Reshape
        ↓
Conv2DTranspose
        ↓
Batch Normalization
        ↓
LeakyReLU
        ↓
Conv2DTranspose
        ↓
Batch Normalization
        ↓
LeakyReLU
        ↓
Conv2DTranspose
        ↓
Tanh
        ↓
28 × 28 × 1 Image
```

The final `tanh` activation produces pixel values in the range `[-1, 1]`.

---

## 2. Discriminator

The Discriminator receives an image and predicts whether it is a real MNIST image or a generated image.

Architecture:

```text
28 × 28 × 1 Image
        ↓
Conv2D
        ↓
LeakyReLU
        ↓
Dropout
        ↓
Conv2D
        ↓
LeakyReLU
        ↓
Dropout
        ↓
Flatten
        ↓
Dense
        ↓
Sigmoid
        ↓
Real / Fake Probability
```

The final sigmoid output produces a probability between 0 and 1.

---

## Loss Function

Binary Cross-Entropy is used for both networks.

### Generator

The Generator tries to make the Discriminator classify generated images as real.

```python
generator_loss = BCE(1, discriminator(fake_images))
```

### Discriminator

The Discriminator learns from both real and generated images.

```python
real_loss = BCE(1, discriminator(real_images))
fake_loss = BCE(0, discriminator(fake_images))

discriminator_loss = real_loss + fake_loss
```

---

## Optimizer

The Adam optimizer is used with the required parameters:

```python
Adam(
    learning_rate=0.0002,
    beta_1=0.5
)
```

These settings are commonly used in GAN training to provide stable optimization.

---

## Training

The GAN is trained for at least 20 epochs.

During each training step:

1. Random noise is generated.
2. The Generator creates fake images.
3. The Discriminator evaluates real MNIST images.
4. The Discriminator evaluates generated images.
5. Generator loss is calculated.
6. Discriminator loss is calculated.
7. Gradients are calculated.
8. Both networks are updated.

The training process is repeated for all batches and epochs.

---

## Generated Images

Generated image samples are saved during training.

The project saves generated images at:

* Epoch 1
* Epoch 5
* Epoch 10
* Epoch 15
* Epoch 20

These images demonstrate how the Generator improves from random/noisy outputs toward recognizable handwritten digits.

---

## Loss Visualization

The project plots:

* Generator Loss
* Discriminator Loss

The loss graph helps analyze adversarial training behavior.

Because GANs involve two competing networks, the losses do not necessarily decrease smoothly like a conventional classification model.

---

## Project Outputs

The repository contains:

```text
outputs/
├── generated_digits_epoch_01.png
├── generated_digits_epoch_05.png
├── generated_digits_epoch_10.png
├── generated_digits_epoch_15.png
├── generated_digits_epoch_20.png
└── loss_curves.png
```

---

## Technologies Used

* Python
* TensorFlow 2.x
* Keras
* NumPy
* Matplotlib
* Jupyter Notebook
* Git
* GitHub

---

## Requirements

Install the required packages:

```bash
pip install tensorflow numpy matplotlib jupyter
```

Or use:

```bash
pip install -r requirements.txt
```

---

## requirements.txt

```text
tensorflow
numpy
matplotlib
jupyter
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Handwritten-Digit-Generation-GAN.git
```

### 2. Open the project

```bash
cd AI-Handwritten-Digit-Generation-GAN
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
AI_Handwritten_Digit_Generation_GAN.ipynb
```

### 6. Run all cells

The notebook will:

* Download MNIST
* Preprocess images
* Build the Generator
* Build the Discriminator
* Train the GAN
* Generate digit images
* Plot losses
* Save trained models

---

## Expected Result

After training, the Generator should produce images that visually resemble handwritten digits from the MNIST dataset.

Early training outputs may contain random patterns or noise. As training progresses, the generated images should become more structured and increasingly resemble handwritten digits.

---

## Learning Outcomes

After completing this project, the following concepts are demonstrated:

* Generative Artificial Intelligence
* GAN architecture
* Adversarial learning
* Neural network optimization
* Image preprocessing
* Convolutional layers
* Transposed convolution
* Batch normalization
* LeakyReLU
* Dropout
* Binary Cross-Entropy
* Adam optimization
* TensorFlow/Keras model development
* Generated-image visualization
* Loss analysis

---

## Future Improvements

Possible improvements include:

* Implement a full DCGAN architecture.
* Train for more epochs.
* Use CIFAR-10 instead of MNIST.
* Implement Conditional GANs.
* Generate specific requested digits.
* Experiment with different latent-space dimensions.
* Add FID or other image-quality metrics.
* Compare different GAN architectures.
* Deploy the Generator using Streamlit.

---

## Conclusion

This project demonstrates how Generative Adversarial Networks can learn the visual distribution of handwritten digits and generate new images from random noise.

The Generator learns to create realistic-looking digit images while the Discriminator learns to distinguish real MNIST images from generated images. Through adversarial training, both networks improve together and the generated samples become increasingly similar to the original dataset.

---

## Author

Simon V

Aspiring Data Scientist | Data Analyst | Machine Learning | Deep Learning | Generative AI

GitHub: `https://github.com/SimonAdvika`

LinkedIn: `www.linkedin.com/in/simon-v-advika`
