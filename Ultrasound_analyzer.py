import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

source_dir = "Datasets"


train_dir = "Datasets/train"
val_dir = "Datasets/val"


classes = ["normal", "benign", "malignant"]

for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    images = os.listdir(os.path.join(source_dir, cls))
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(source_dir, cls, img), os.path.join(train_dir, cls, img))
   for img in val_imgs:
        shutil.copy(os.path.join(source_dir, cls, img), os.path.join(val_dir, cls, img))

train_dir = 'Datasets/train'
val_dir = 'Datasets/val'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
model.save("ultrasound_classifier.h5")


# Training Details
# Why Epochs Matter
# When training a model:
  # You feed the entire dataset to the model once → that's 1 epoch.
  # Typically, you need multiple epochs so the model can learn patterns from the data more effectively.
  # During each epoch, the model updates its weights using optimization algorithms like gradient descent.
# Example:
  # If you have a dataset with 10,000 images and you train the model for 5 epochs, the model will see all 10,000 images five times during training.


# Epoch 1/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6879 - loss: 0.8575C:\python 3.9\lib\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
#   self._warn_if_super_not_called()
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 106s 1s/step - accuracy: 0.6884 - loss: 0.8552 - val_accuracy: 0.7714 - val_loss: 0.5881
# Epoch 2/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 98s 1s/step - accuracy: 0.7598 - loss: 0.5764 - val_accuracy: 0.7748 - val_loss: 0.5491
# Epoch 3/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 113s 2s/step - accuracy: 0.7565 - loss: 0.5676 - val_accuracy: 0.7765 - val_loss: 0.5453
# Epoch 4/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 96s 1s/step - accuracy: 0.7743 - loss: 0.5674 - val_accuracy: 0.7782 - val_loss: 0.5453
# Epoch 5/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 93s 1s/step - accuracy: 0.7680 - loss: 0.5305 - val_accuracy: 0.7798 - val_loss: 0.5497
# Epoch 6/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 92s 1s/step - accuracy: 0.7734 - loss: 0.5312 - val_accuracy: 0.7529 - val_loss: 0.5774
# Epoch 7/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 93s 1s/step - accuracy: 0.7788 - loss: 0.5322 - val_accuracy: 0.7597 - val_loss: 0.5867
# Epoch 8/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 93s 1s/step - accuracy: 0.7531 - loss: 0.5543 - val_accuracy: 0.7765 - val_loss: 0.5384
# Epoch 9/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 93s 1s/step - accuracy: 0.7711 - loss: 0.5299 - val_accuracy: 0.7765 - val_loss: 0.5554
# Epoch 10/10
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 92s 1s/step - accuracy: 0.7686 - loss: 0.5331 - val_accuracy: 0.7765 - val_loss: 0.5504
