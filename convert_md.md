
```python

    optimizer = AlexSGD(learning_rate, momentum, weight_decay)
    accuracy = CategoricalAccuracy()

    def train_step(images, labels):

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            losses_in_batch = SparseCategoricalCrossentropy(labels, predictions)

        gradients = tape.gradient(losses_in_batch)
        optimizer.apply_gradients(gradients)

        Calculate_mean_loss(losses_in_batch)
        accuracy.update(labels, predictions)

    def test_step(images, labels):
        predictions = model(images, training =False)
        losses_in_batch = SparseCategoricalCrossentropy(labels, predictions)
        
        Calculate_test_mean_loss(losses_in_batch)
        accuracy.update(labels, predictions)
```