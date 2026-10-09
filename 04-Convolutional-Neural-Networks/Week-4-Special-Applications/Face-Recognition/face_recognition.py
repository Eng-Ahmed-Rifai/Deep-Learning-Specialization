"""
Face Recognition & Verification with Siamese Networks (FaceNet)
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 4, Lab 1)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this lab, we implement:
1. triplet_loss: Vectorized Triplet Loss function in TensorFlow comparing Anchor, Positive, and Negative.
2. verify: 1-to-1 face verification against authorized identities (with image_path, identity, database, model).
3. who_is_it: 1-to-K face recognition across an organization identity database (with image_path, database, model).
"""

import tensorflow as tf
import numpy as np


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula:
    L(A, P, N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha, 0)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    alpha -- float, the margin parameter
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    
    # Step 3: Subtract the two distances and add alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss


def img_to_encoding(image_path, model):
    """
    Transforms image at image_path into 128-dimensional normalized embedding using model.
    """
    if isinstance(image_path, str):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
    else:
        x_train = image_path
        if x_train.ndim == 3:
            x_train = np.expand_dims(x_train, axis=0)
            
    if hasattr(model, "predict_on_batch"):
        embedding = model.predict_on_batch(x_train)
    elif callable(model):
        embedding = model(x_train)
    else:
        embedding = x_train
        
    embedding = np.array(embedding).flatten()
    norm = np.linalg.norm(embedding, ord=2)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def verify(image_path, identity, database, model=None):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
        image_path -- path to an image or pre-computed embedding vector
        identity -- string, name of the person you'd like to verify the identity
        database -- python dictionary mapping names to their corresponding 128-d encodings
        model -- your Inception/FaceNet model instance in Keras
    
    Returns:
        dist -- distance between the image_path encoding and the identity encoding in the database
        door_open -- True, if the door should open. False otherwise.
    """
    # Step 1: Compute the encoding for the image from image_path
    if isinstance(image_path, (np.ndarray, tf.Tensor)) and np.array(image_path).ndim == 1:
        encoding = np.array(image_path)
    else:
        encoding = img_to_encoding(image_path, model)
        
    # Step 2: Compute distance with identity's image
    dist = float(np.linalg.norm(encoding - database[identity]))
    
    # Step 3: Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        door_open = True
    else:
        door_open = False
        
    return dist, door_open


def who_is_it(image_path, database, model=None):
    """
    Implements 1:K face recognition for a given face image.
    
    Arguments:
        image_path -- path to an image or pre-computed embedding vector
        database -- database containing image encodings along with the corresponding identity's name
        model -- your Inception/FaceNet model instance in Keras
    
    Returns:
        min_dist -- the minimum distance between image_path's encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """
    # Step 1: Compute the target "encoding" for the image
    if isinstance(image_path, (np.ndarray, tf.Tensor)) and np.array(image_path).ndim == 1:
        encoding = np.array(image_path)
    else:
        encoding = img_to_encoding(image_path, model)
        
    # Initialize "min_dist" to a large value
    min_dist = float("inf")
    identity = "Not in the database."
    
    # Loop over the database dictionary's names and encodings
    for (name, db_enc) in database.items():
        dist = float(np.linalg.norm(encoding - db_enc))
        if dist < min_dist:
            min_dist = dist
            identity = name
            
    if min_dist > 0.7:
        identity = "Not in the database."
        
    return min_dist, identity


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: Face Recognition & Triplet Loss (Coursera C4W4 Lab 1)")
    print("=" * 70)
    
    tf.random.set_seed(1)
    np.random.seed(1)
    
    # 1. Test triplet_loss with Coursera reference values
    y_true = (None, None, None)
    y_pred = (
        tf.random.normal([3, 128], mean=6, stddev=0.1, seed=1),
        tf.random.normal([3, 128], mean=1, stddev=1, seed=1),
        tf.random.normal([3, 128], mean=3, stddev=4, seed=1)
    )
    loss = triplet_loss(y_true, y_pred, alpha=0.2)
    # Coursera benchmark value: 527.2598
    assert np.isclose(loss.numpy(), 527.2598, atol=1e-3), f"Loss mismatch: {loss.numpy()}"
    print(f"[PASS] triplet_loss computed: {loss.numpy():.4f} (matches Coursera benchmark 527.2598)")
    
    # Test zero loss bound
    y_pred_zero = (tf.ones([2, 128]), tf.ones([2, 128]), tf.fill([2, 128], 100.0))
    assert triplet_loss(y_true, y_pred_zero, alpha=0.2).numpy() == 0.0, "Zero loss condition failed"
    print("[PASS] triplet_loss margin zero-bound verified.")
    
    # 2. Database setup
    db = {
        "ahmed": np.random.randn(128),
        "kiron": np.random.randn(128),
        "deeplearning": np.random.randn(128)
    }
    # Normalize database embeddings
    for k in db:
        db[k] = db[k] / np.linalg.norm(db[k])
        
    # Mock model that maps dummy input to target embedding
    class MockFaceModel:
        def __init__(self, target_vec):
            self.target_vec = target_vec / np.linalg.norm(target_vec)
        def predict_on_batch(self, x):
            return np.expand_dims(self.target_vec, axis=0)
            
    model_ahmed = MockFaceModel(db["ahmed"])
    model_impostor = MockFaceModel(np.random.randn(128) * 10)
    
    # Test verify(image_path, identity, database, model)
    dummy_input_img = np.ones((160, 160, 3), dtype=np.float32)
    dist, verified = verify(dummy_input_img, "ahmed", db, model=model_ahmed)
    assert verified is True, "Authorized identity must verify as True"
    assert np.isclose(dist, 0.0, atol=1e-5), f"Expected distance ~0.0, got {dist}"
    print(f"[PASS] verify(image_path, 'ahmed', db, model) passed: dist={dist:.4f}, door_open={verified}")
    
    dist_imp, verified_imp = verify(dummy_input_img, "ahmed", db, model=model_impostor)
    assert verified_imp is False, "Impostor must be rejected"
    print(f"[PASS] verify rejected impostor: dist={dist_imp:.4f}, door_open={verified_imp}")
    
    # Test who_is_it(image_path, database, model)
    model_kiron = MockFaceModel(db["kiron"])
    min_dist, pred_id = who_is_it(dummy_input_img, db, model=model_kiron)
    assert pred_id == "kiron", f"Expected 'kiron', got {pred_id}"
    assert np.isclose(min_dist, 0.0, atol=1e-5)
    print(f"[PASS] who_is_it(image_path, db, model) successfully recognized: {pred_id} (dist: {min_dist:.4f})")
    
    # Test who_is_it for unknown person
    min_dist_unk, pred_id_unk = who_is_it(dummy_input_img, db, model=model_impostor)
    assert pred_id_unk == "Not in the database."
    print(f"[PASS] who_is_it correctly handles unknown person: {pred_id_unk}")
    
    print("=" * 70)
    print("ALL FACE RECOGNITION UNIT TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)
