# Alexnet

Implemented "ImageNet Classification with Deep Convolutional Neural Networks (NIPS 2012)" a.k.a. Alexnet using tensorflow2.

# Note

- I have implemented their customized nesterov gradient descent optimizer, but do not design distribution learning.

- As shown below, the top-1 validation error using 1-CNN model achieved 39.98, similar to that described in Table 2 of the paper (40.7).

    ![top1](figure/top1.png)

- Top 5 validation error acheived 5.77 because I have used only 10 classes to train and validate the model.
    ![top5](figure/top5.png)

# DataGeneration

1. Download ILSVR2012 dataset.
2. Run the setup shell script file that can download from [here](https://github.com/ShinYwings/setup-imgnet-dataset)

    ```
    zsh image_val_setup.sh
    ```

3. Convert imagenet dataset to tfrecord

    ```
    python ImageNetDataset.py --traindir "\train\dir\abs\path" --valdir "\val\dir\abs\path" --output_path "output/dir/name"
    ```

# Training and validation

```
python main.py --traindir "tfrecord_train_dir_abs_path" --valdir "tfrecord_val_dir_abs_path"
```
