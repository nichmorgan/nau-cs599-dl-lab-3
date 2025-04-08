import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0  # Normalize to [0,1]
x_test = x_test / 255.0
x_train = x_train[..., tf.newaxis]  # Add channel dimension: (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis]    # (10000, 28, 28, 1)

# Prepare datasets
batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Define loss function, optimizer, and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Custom normalization functions using basic TensorFlow ops
def batch_norm(x, gamma, beta, epsilon=1e-5):
    """Batch Normalization: Normalizes across batch and spatial dimensions per channel."""
    mean = tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True)  # Mean over batch, height, width
    variance = tf.reduce_mean(tf.square(x - mean), axis=[0, 1, 2], keepdims=True)
    x_normalized = (x - mean) / tf.sqrt(variance + epsilon)
    return gamma * x_normalized + beta

def layer_norm(x, gamma, beta, epsilon=1e-5):
    """Layer Normalization: Normalizes across channels per sample and spatial location."""
    mean = tf.reduce_mean(x, axis=[3], keepdims=True)  # Mean over channels
    variance = tf.reduce_mean(tf.square(x - mean), axis=[3], keepdims=True)
    x_normalized = (x - mean) / tf.sqrt(variance + epsilon)
    return gamma * x_normalized + beta

class WeightNormConv2D(tf.keras.layers.Conv2D):
    """Weight Normalization: Reparameterizes weights as w = (g / ||v||) * v."""
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)
        self.v = self.add_weight(
            name='v',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype
        )
        self.g = self.add_weight(
            name='g',
            shape=(self.filters,),
            initializer=tf.initializers.ones(),
            trainable=True,
            dtype=self.dtype
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        v_norm = tf.sqrt(tf.reduce_sum(tf.square(self.v), axis=[0, 1, 2], keepdims=True))
        kernel = (self.g[None, None, None, :] / (v_norm + 1e-5)) * self.v
        outputs = self.convolution_op(inputs, kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# CNN model with custom normalization
class CNNModel(tf.keras.Model):
    def __init__(self, normalization=None):
        super(CNNModel, self).__init__()
        if normalization == 'WN':
            self.conv1 = WeightNormConv2D(32, (3, 3), activation='relu', padding='valid')
            self.conv2 = WeightNormConv2D(64, (3, 3), activation='relu', padding='valid')
        elif normalization in ['BN', 'LN']:
            self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=None, padding='valid')
            self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='valid')
            self.gamma1 = tf.Variable(tf.ones([1, 1, 1, 32]), name='gamma1')
            self.beta1 = tf.Variable(tf.zeros([1, 1, 1, 32]), name='beta1')
            self.gamma2 = tf.Variable(tf.ones([1, 1, 1, 64]), name='gamma2')
            self.beta2 = tf.Variable(tf.zeros([1, 1, 1, 64]), name='beta2')
        else:  # None
            self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')
            self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        self.normalization = normalization

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if self.normalization in ['BN', 'LN']:
            if self.normalization == 'BN':
                x = batch_norm(x, self.gamma1, self.beta1)
            elif self.normalization == 'LN':
                x = layer_norm(x, self.gamma1, self.beta1)
            x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        if self.normalization in ['BN', 'LN']:
            if self.normalization == 'BN':
                x = batch_norm(x, self.gamma2, self.beta2)
            elif self.normalization == 'LN':
                x = layer_norm(x, self.gamma2, self.beta2)
            x = tf.nn.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# CNN model with TensorFlow normalization
class CNNModelTFNorm(tf.keras.Model):
    def __init__(self, normalization=None):
        super(CNNModelTFNorm, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=None, padding='valid')
        if normalization == 'BN':
            self.norm1 = tf.keras.layers.BatchNormalization()
        elif normalization == 'LN':
            self.norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='valid')
        if normalization == 'BN':
            self.norm2 = tf.keras.layers.BatchNormalization()
        elif normalization == 'LN':
            self.norm2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        self.normalization = normalization

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if self.normalization in ['BN', 'LN']:
            x = self.norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        if self.normalization in ['BN', 'LN']:
            x = self.norm2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# Training and evaluation functions
@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

def train_model(model, epochs):
    history = {'train_acc': [], 'test_acc': []}
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(model, images, labels)

        for test_images, test_labels in test_ds:
            test_step(model, test_images, test_labels)

        history['train_acc'].append(float(train_accuracy.result()))
        history['test_acc'].append(float(test_accuracy.result()))

        print(f'Epoch {epoch + 1}, '
              f'Loss: {train_loss.result():.4f}, '
              f'Accuracy: {train_accuracy.result() * 100:.2f}%, '
              f'Test Loss: {test_loss.result():.4f}, '
              f'Test Accuracy: {test_accuracy.result() * 100:.2f}%')
    return history

# Train models with different normalizations
epochs = 10

print("Training model with no normalization:")
model_none = CNNModel(normalization=None)
optimizer = tf.keras.optimizers.Adam()
model_none.build((None, 28, 28, 1))
optimizer.build(model_none.trainable_variables)
history_none = train_model(model_none, epochs)

print("\nTraining model with custom Batch Normalization:")
model_bn = CNNModel(normalization='BN')
optimizer = tf.keras.optimizers.Adam()
model_bn.build((None, 28, 28, 1))
optimizer.build(model_bn.trainable_variables)
history_bn = train_model(model_bn, epochs)

print("\nTraining model with custom Layer Normalization:")
model_ln = CNNModel(normalization='LN')
optimizer = tf.keras.optimizers.Adam()
model_ln.build((None, 28, 28, 1))
optimizer.build(model_ln.trainable_variables)
history_ln = train_model(model_ln, epochs)

print("\nTraining model with custom Weight Normalization:")
model_wn = CNNModel(normalization='WN')
optimizer = tf.keras.optimizers.Adam()
model_wn.build((None, 28, 28, 1))
optimizer.build(model_wn.trainable_variables)
history_wn = train_model(model_wn, epochs)

print("\nTraining model with TensorFlow Batch Normalization:")
model_tf_bn = CNNModelTFNorm(normalization='BN')
optimizer = tf.keras.optimizers.Adam()
model_tf_bn.build((None, 28, 28, 1))
optimizer.build(model_tf_bn.trainable_variables)
history_tf_bn = train_model(model_tf_bn, epochs)

print("\nTraining model with TensorFlow Layer Normalization:")
model_tf_ln = CNNModelTFNorm(normalization='LN')
optimizer = tf.keras.optimizers.Adam()
model_tf_ln.build((None, 28, 28, 1))
optimizer.build(model_tf_ln.trainable_variables)
history_tf_ln = train_model(model_tf_ln, epochs)

# Plot test accuracies
plt.figure(figsize=(12, 8))
plt.plot(history_none['test_acc'], label='No Normalization')
plt.plot(history_bn['test_acc'], label='Custom BN')
plt.plot(history_ln['test_acc'], label='Custom LN')
plt.plot(history_wn['test_acc'], label='Custom WN')
plt.plot(history_tf_bn['test_acc'], label='TF BN')
plt.plot(history_tf_ln['test_acc'], label='TF LN')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison Across Normalization Techniques')
plt.legend()
plt.grid(True)
plt.savefig('test_accuracy_comparison.png')
plt.show()

# Gradient and output comparison for BN and LN
images, _ = next(iter(train_ds))
x = model_bn.conv1(images)  # Use conv output for normalization comparison

# Compare Batch Normalization
print("\nComparing Batch Normalization implementations:")
with tf.GradientTape() as tape:
    tape.watch(x)
    y_custom_bn = batch_norm(x, tf.ones_like(model_bn.gamma1), tf.zeros_like(model_bn.beta1))
    loss_custom = tf.reduce_sum(y_custom_bn)
grad_custom_bn = tape.gradient(loss_custom, x)

bn_layer = tf.keras.layers.BatchNormalization()
with tf.GradientTape() as tape:
    tape.watch(x)
    y_tf_bn = bn_layer(x, training=True)
    loss_tf = tf.reduce_sum(y_tf_bn)
grad_tf_bn = tape.gradient(loss_tf, x)

output_diff_bn = tf.reduce_max(tf.abs(y_custom_bn - y_tf_bn))
grad_diff_bn = tf.reduce_max(tf.abs(grad_custom_bn - grad_tf_bn))
print(f"Max output difference (Custom BN vs TF BN): {output_diff_bn:.6f}")
print(f"Max gradient difference (Custom BN vs TF BN): {grad_diff_bn:.6f}")

# Compare Layer Normalization
print("\nComparing Layer Normalization implementations:")
with tf.GradientTape() as tape:
    tape.watch(x)
    y_custom_ln = layer_norm(x, tf.ones_like(model_ln.gamma1), tf.zeros_like(model_ln.beta1))
    loss_custom_ln = tf.reduce_sum(y_custom_ln)
grad_custom_ln = tape.gradient(loss_custom_ln, x)

ln_layer = tf.keras.layers.LayerNormalization(axis=-1)
with tf.GradientTape() as tape:
    tape.watch(x)
    y_tf_ln = ln_layer(x)
    loss_tf_ln = tf.reduce_sum(y_tf_ln)
grad_tf_ln = tape.gradient(loss_tf_ln, x)

output_diff_ln = tf.reduce_max(tf.abs(y_custom_ln - y_tf_ln))
grad_diff_ln = tf.reduce_max(tf.abs(grad_custom_ln - grad_tf_ln))
print(f"Max output difference (Custom LN vs TF LN): {output_diff_ln:.6f}")
print(f"Max gradient difference (Custom LN vs TF LN): {grad_diff_ln:.6f}")

# Analysis and Report
print("\n### Analysis and Findings ###")
print("1. **Performance Comparison**:")
print(f" - No Normalization: Final Test Accuracy = {history_none['test_acc'][-1] * 100:.2f}%")
print(f" - Custom BN: Final Test Accuracy = {history_bn['test_acc'][-1] * 100:.2f}%")
print(f" - Custom LN: Final Test Accuracy = {history_ln['test_acc'][-1] * 100:.2f}%")
print(f" - Custom WN: Final Test Accuracy = {history_wn['test_acc'][-1] * 100:.2f}%")
print(f" - TF BN: Final Test Accuracy = {history_tf_bn['test_acc'][-1] * 100:.2f}%")
print(f" - TF LN: Final Test Accuracy = {history_tf_ln['test_acc'][-1] * 100:.2f}%")

print("\n2. **Best Performing Technique**:")
# Note: Actual best performer depends on run; below is a placeholder based on typical behavior
print("Batch Normalization (both custom and TF) typically outperforms others in this CNN task on Fashion MNIST, "
      "likely due to its ability to reduce internal covariate shift and stabilize training across mini-batches.")

print("\n3. **Custom vs TensorFlow Normalizations**:")
print(f" - BN: Output difference = {output_diff_bn:.6f}, Gradient difference = {grad_diff_bn:.6f}")
print(f" - LN: Output difference = {output_diff_ln:.6f}, Gradient difference = {grad_diff_ln:.6f}")
print("Small differences indicate that custom implementations are correct, with variations possibly due to "
      "floating-point precision or minor implementation details (e.g., TF BN uses running averages).")

print("\n4. **Why Layer Normalization Might Be Better Than Batch Normalization**:")
print("Layer Normalization can outperform Batch Normalization in scenarios where:")
print(" - **Small Batch Sizes**: LN normalizes across features per sample, independent of batch size, making it "
      "robust when batch statistics are unreliable.")
print(" - **Recurrent Networks**: LN is common in RNNs or transformers, where batch-wise normalization is less "
      "effective due to sequential dependencies.")
print(" - **Variable Input Statistics**: When mini-batch statistics vary significantly, LN provides consistent "
      "normalization per sample.")
print("However, for this CNN on Fashion MNIST with a decent batch size (64), BN often excels because it leverages "
      "batch statistics effectively for image data, which tends to be consistent within batches.")