import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
# Try to import XGBoost, if available
try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False
    print("XGBoost not available, will skip XGBoost classifier")

def extract_features(keypoints):
    features = []
    
    # Check dimensions
    if len(keypoints.shape) != 4:
        # Return zeros if unexpected shape
        return np.zeros(50)
    
    M, T, V, C = keypoints.shape  # People, Frames, Joints, Coordinates
    
    # 1. Basic statistics of joint positions
    mean_pos = np.mean(keypoints, axis=(0, 1))  # Average position of each joint
    features.extend(mean_pos.flatten())
    
    # 2. Hand movement features
    hand_joints = [9, 10]  # Assuming COCO format where these are hand indices
    if T > 1:  # Need at least 2 frames for velocity
        # Calculate velocities for hands
        hand_vel = np.diff(keypoints[:, :, hand_joints, :], axis=1)
        mean_hand_vel = np.mean(np.abs(hand_vel), axis=(0, 1))
        features.extend(mean_hand_vel.flatten())
        
        # Max velocity (for impact detection)
        max_hand_vel = np.max(np.abs(hand_vel), axis=(0, 1))
        features.extend(max_hand_vel.flatten())
    
    # 3. Distance between hands and head/torso
    head_idx = 0  # Nose in COCO format
    torso_idx = [5, 6]  # Shoulders in COCO format
    
    for p in range(M):  # For each person
        for t in range(T):  # For each frame
            # Head to hands distance
            for h in hand_joints:
                if np.sum(keypoints[p, t, h]) > 0:  # If hand is visible
                    head_hand = np.linalg.norm(keypoints[p, t, h] - keypoints[p, t, head_idx])
                    features.append(head_hand)
            
            # Shoulders to hands
            for s in torso_idx:
                for h in hand_joints:
                    if np.sum(keypoints[p, t, h]) > 0 and np.sum(keypoints[p, t, s]) > 0:
                        shoulder_hand = np.linalg.norm(keypoints[p, t, h] - keypoints[p, t, s])
                        features.append(shoulder_hand)
    
    # Normalize and pad/truncate to fixed length
    features = np.array(features, dtype=np.float32)
    if len(features) > 100:
        return features[:100]  # Take first 100
    else:
        return np.pad(features, (0, 100 - len(features)))  # Pad to 100

def main():
    # Define classifiers with the specified parameters
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Add XGBoost if available with max_depth set to 6
    if has_xgboost:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=6, 
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    
    # Define action names for reports
    action_names = [
        'Head with the left hand',
        'Head with the right hand',
        'Torso with the left hand',
        'Torso with the right hand',
        'Block with the left hand',
        'Block with the right hand',
        'Miss with the left hand',
        'Miss with the right hand'
    ]
    
    # Load data
    print("Loading data...")
    # Training on balanced data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'boxing_balanced.pkl')
    with open(file_path, 'rb') as f:
        balanced_data = pickle.load(f)

    # Get training data from balanced dataset
    train_videos = balanced_data['split']['xsub_train']
    train_data = [ann for ann in balanced_data['annotations'] if ann['frame_dir'] in train_videos]
    
    # Get test data from original dataset
    test_videos = balanced_data['split']['xsub_test']
    test_data = [ann for ann in balanced_data['annotations'] if ann['frame_dir'] in test_videos]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Extract features
    print("Extracting features...")
    X_train = []
    y_train = []
    
    start_time = time.time()
    for i, item in enumerate(train_data):
        try:
            features = extract_features(item['keypoint'])
            X_train.append(features)
            y_train.append(item['label'])
            
            if (i+1) % 500 == 0:
                print(f"Processed {i+1}/{len(train_data)} training samples")
        except Exception as e:
            print(f"Error processing training sample {i}: {e}")
    
    X_test = []
    y_test = []
    
    for i, item in enumerate(test_data):
        try:
            features = extract_features(item['keypoint'])
            X_test.append(features)
            y_test.append(item['label'])
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(test_data)} test samples")
        except Exception as e:
            print(f"Error processing test sample {i}: {e}")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    feature_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_time:.2f} seconds")
    print(f"Feature vectors shape: {X_train.shape}")
    
    # Check for class distribution in test set
    class_counts = {}
    for label in y_test:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    print("Test set class distribution:")
    for label, count in sorted(class_counts.items()):
        percent = count / len(y_test) * 100
        print(f"  Class {label} ({action_names[label]}): {count} samples ({percent:.2f}%)")
    
    # Set up 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train, cross-validate, and evaluate all classifiers
    results = {}
    best_accuracy = 0
    best_classifier_name = None
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Perform 5-fold cross-validation on the training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
        print(f"{name} 5-fold CV Accuracy: Mean = {np.mean(cv_scores):.4f}, Std = {np.std(cv_scores):.4f}")
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"{name} trained in {train_time:.2f} seconds")
        
        # Predict on test set
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} test accuracy: {accuracy:.4f}")
        
        # Save detailed report
        report = classification_report(y_test, y_pred, target_names=action_names)
        print(f"\nClassification Report for {name}:")
        print(report)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=action_names, yticklabels=action_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name}.png')
        plt.close()
        
        # Save results
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'cv_scores': cv_scores
        }
        
        # Save the model
        joblib.dump(clf, f'boxing_classifier_{name}.joblib')
        
        # Track best classifier
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier_name = name

    
    # Print summary and best classifier
    print("\n===== SUMMARY =====")
    for name, result in results.items():
        print(f"{name}: Test Accuracy = {result['accuracy']:.4f} | CV Mean = {np.mean(result['cv_scores']):.4f}")
    
    print(f"\nBest classifier: {best_classifier_name} with test accuracy {best_accuracy:.4f}")
    print(f"The best model is saved as boxing_classifier_{best_classifier_name}.joblib")

if __name__ == "__main__":
    main()
