```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,
                                     momentum=0.9,
                                        nesterov=False)

with tf.device('/gpu:1'):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, lrn_info, NUM_CLASSES)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, _model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, _model.trainable_variables))

    for epoch in range(epochs=90):
        for images, labels in train_ds:
            train_step(images, labels)
```