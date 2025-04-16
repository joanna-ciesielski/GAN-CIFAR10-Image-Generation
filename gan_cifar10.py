import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Dropout,
                                     BatchNormalization, Activation,
                                     LeakyReLU, UpSampling2D, Conv2D)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

###############################################################################
# 1. Hyperparameters
###############################################################################
image_shape = (32, 32, 3)
latent_dimensions = 100

###############################################################################
# 2. Build the Generator
###############################################################################
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.7))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.7))
    model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dimensions,))
    image = model(noise)
    return Model(noise, image)

###############################################################################
# 3. Build the Discriminator
###############################################################################
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
                     padding="same", input_shape=image_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.7))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.7))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.7))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)
    return Model(image, validity)

###############################################################################
# 4. Utility function to display generated images
###############################################################################
def display_images(epoch, generator, rows=4, cols=4):
    noise = np.random.normal(0, 1, (rows * cols, latent_dimensions))
    generated_images = generator.predict(noise)

    # Rescale from [-1, 1] to [0, 1]
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
    count = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis('off')
            count += 1

    plt.suptitle(f"Generated Images at Epoch {epoch}", fontsize=14)
    plt.show()

###############################################################################
# 5. Load CIFAR-10 data
###############################################################################
(X, y), (_, _) = tf.keras.datasets.cifar10.load_data()
# Filter to a single class, e.g., label '8'
X = X[y.flatten() == 8]

# Normalize to [-1, 1]
X = (X / 127.5) - 1.0

###############################################################################
# 6. Build and compile networks
###############################################################################
discriminator = build_discriminator()
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy']
)

generator = build_generator()

# Combined network (generator + frozen discriminator)
discriminator.trainable = False
z = Input(shape=(latent_dimensions,))
img = generator(z)
valid = discriminator(img)
combined_network = Model(z, valid)
combined_network.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5)
)

###############################################################################
# 7. Train the GAN
###############################################################################
num_epochs = 15000
batch_size = 32  # Updated to 32

# Real = 1, Fake = 0
valid_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(num_epochs):
    # -------------------------
    #  Train the Discriminator
    # -------------------------
    discriminator.trainable = True

    # 1) Real images
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_imgs = X[idx]
    d_loss_real = discriminator.train_on_batch(real_imgs, valid_labels)

    # 2) Fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    fake_imgs = generator.predict(noise)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

    # Combine real+fake results
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
    d_acc = 0.5 * (d_loss_real[1] + d_loss_fake[1])

    # ---------------------
    #  Train the Generator
    # ---------------------
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    g_loss_val = combined_network.train_on_batch(noise, valid_labels)
    if isinstance(g_loss_val, (list, tuple)):
        g_loss_val = g_loss_val[0]

    # Print / Display images
    # Show images at epoch 0, then every 2500 epochs
    if epoch == 0:
        print(f"\nEpoch {epoch}:")
        print(f"Discriminator Loss: {d_loss:.4f}, Accuracy: {d_acc:.4f}")
        print(f"Generator Loss: {g_loss_val:.4f}")
        display_images(epoch, generator)

    if (epoch + 1) % 2500 == 0:
        print(f"\nEpoch {epoch + 1}:")
        print(f"Discriminator Loss: {d_loss:.4f}, Accuracy: {d_acc:.4f}")
        print(f"Generator Loss: {g_loss_val:.4f}")
        display_images(epoch + 1, generator)

# Final epoch images
print("\nFinal Epoch Images (Epoch 15000):")
display_images(num_epochs, generator)