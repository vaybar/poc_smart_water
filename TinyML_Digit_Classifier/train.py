"""
train.py - Micro-MobileNet Model Training Script

Trains the Micro-MobileNet model on digit dataset, applies data augmentation,
learning rate schedules, and saves the trained float32 model checkpoint.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from model_builder import build_micro_mobilenet
from dataset import load_digit_dataset, get_data_augmentation

def train_model(alpha=config.ALPHA, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
    print("=" * 60)
    print(f"  Training Micro-MobileNet (Alpha={alpha}, Input={config.INPUT_SHAPE})")
    print("=" * 60)
    
    # 1. Load Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_digit_dataset()
    
    # Compute class weights for dataset balance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights computed for {len(classes)} classes.")
    
    # 2. Build Model
    model = build_micro_mobilenet(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        alpha=alpha,
        dropout_rate=0.2
    )
    
    # Wrap model with Data Augmentation
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="aug_input")
    augmented = get_data_augmentation()(inputs)
    outputs = model(augmented)
    full_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MicroMobileNet_Train")
    
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.FLOAT_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 3. Fit Model
    print(f"Starting training for {epochs} epochs...")
    history = full_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Evaluate base model without augmentation wrapper
    print("\nEvaluating base model on test set...")
    base_eval_model = build_micro_mobilenet(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        alpha=alpha,
        dropout_rate=0.0
    )
    # Save weights from full model's internal model layer
    model.save_weights("temp_weights.weights.h5")
    base_eval_model.load_weights("temp_weights.weights.h5")
    if os.path.exists("temp_weights.weights.h5"):
        os.remove("temp_weights.weights.h5")
        
    base_eval_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    test_loss, test_acc = base_eval_model.evaluate(x_test, y_test, verbose=0)
    print(f"--> Base Test Loss: {test_loss:.4f}, Base Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save base model cleanly
    base_eval_model.save(str(config.FLOAT_MODEL_PATH))
    print(f"Saved Float32 model to: {config.FLOAT_MODEL_PATH}")
    
    return base_eval_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Micro-MobileNet for TinyML Digit Classifier")
    parser.add_argument("--alpha", type=float, default=config.ALPHA, help="Width multiplier (0.10, 0.25, 0.35, 0.50)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    args = parser.parse_args()
    
    train_model(alpha=args.alpha, epochs=args.epochs)
