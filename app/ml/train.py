"""Training Script — FER-2013 CNN training."""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description='Train the facial expression recognition CNN on FER-2013.')
    parser.add_argument('--data', required=True, help='Path to fer2013.csv')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--output', default='models/fer2013_cnn.h5', help='Output path for the trained model')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--model-type', choices=['baseline', 'challenger', 'low_latency', 'hybrid'], default='baseline',
                        help='Architecture type: baseline (Phase 5.5 CNN), challenger (Phase 5.6 Residual CNN), low_latency (Phase 5.7 Depthwise-Separable CNN), or hybrid (Phase 5.8 Inverted Residual Hybrid CNN)')
    parser.add_argument('--augment', action='store_true', help='Enable real-time training data augmentation')
    parser.add_argument('--patience', type=int, default=12, help='Patience for EarlyStopping')
    args = parser.parse_args()

    # Automatically set model-type based on output filename if default
    if 'low_latency' in args.output and args.model_type == 'baseline':
        args.model_type = 'low_latency'
    elif 'challenger' in args.output and args.model_type == 'baseline':
        args.model_type = 'challenger'
    elif 'hybrid' in args.output and args.model_type == 'baseline':
        args.model_type = 'hybrid'

    try:
        from app.ml.preprocessor import parse_fer2013
        from app.ml.model import build_model, build_challenger_model, build_low_latency_model, build_hybrid_model
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
        
        # Enforce deterministic random seeds
        tf.keras.utils.set_random_seed(42)
        np.random.seed(42)
        
        # Centralized class mapping
        from app.models.emotion import EMOTION_CLASSES
        
        print(f"Parsing dataset from {args.data}...")
        splits = parse_fer2013(args.data)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_public, y_public = splits['public_test']
        X_private, y_private = splits['private_test']
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, PublicTest: {X_public.shape}, PrivateTest: {X_private.shape}")
        
        # Report class distribution
        y_train_indices = np.argmax(y_train, axis=1)
        unique, counts = np.unique(y_train_indices, return_counts=True)
        print("Training class distribution:")
        for cls_idx, count in zip(unique, counts):
            print(f"  {EMOTION_CLASSES[cls_idx]}: {count}")
            
        # Compute class weights for imbalance
        weights = compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)
        class_weight_dict = {i: w for i, w in enumerate(weights)}
        print("Computed class weights:", class_weight_dict)
        
        # Build model based on model-type
        print(f"\nBuilding model type: '{args.model_type}'")
        if args.model_type == 'challenger':
            model = build_challenger_model()
        elif args.model_type == 'low_latency':
            model = build_low_latency_model()
        elif args.model_type == 'hybrid':
            model = build_hybrid_model()
        else:
            model = build_model()
            
        model.summary()
        
        # Ensure output directory exists before training
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=args.output,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        if args.model_type == 'challenger' or args.epochs > 10:
            callbacks.extend([
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=args.patience,
                    restore_best_weights=True,
                    verbose=1
                )
            ])
        
        # Train with or without augmentation
        if args.augment:
            print("Training with real-time data augmentation...")
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.10,
                height_shift_range=0.10,
                zoom_range=0.10,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            train_flow = datagen.flow(X_train, y_train, batch_size=args.batch_size)
            model.fit(
                train_flow,
                steps_per_epoch=len(X_train) // args.batch_size,
                epochs=args.epochs,
                validation_data=(X_val, y_val),
                class_weight=class_weight_dict,
                callbacks=callbacks
            )
        else:
            model.fit(
                X_train, y_train, 
                batch_size=args.batch_size, 
                epochs=args.epochs, 
                validation_data=(X_val, y_val),
                class_weight=class_weight_dict,
                callbacks=callbacks
            )
        
        # Load the best model weights saved by ModelCheckpoint for evaluation
        if os.path.exists(args.output):
            model.load_weights(args.output)
        
        # Evaluation Helper
        def evaluate_split(name, X, y):
            print(f"\n================ {name} EVALUATION ================")
            if len(X) == 0:
                print("No data available for this split.")
                return
            preds = model.predict(X, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            y_true = np.argmax(y, axis=1)
            
            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
            
            print(f"Accuracy: {acc:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}\n")
            print("--- Classification Report ---")
            print(classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, digits=4))
            print("--- Confusion Matrix ---")
            print(confusion_matrix(y_true, y_pred))
            
        evaluate_split("VALIDATION (10% Split)", X_val, y_val)
        evaluate_split("PUBLIC TEST", X_public, y_public)
        evaluate_split("PRIVATE TEST", X_private, y_private)
        
        if os.path.exists(args.output):
            size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"\nFinal model weights safely present at {args.output} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



