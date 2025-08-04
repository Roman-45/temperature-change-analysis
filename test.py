try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} is installed")
    print(f"✅ Keras {tf.keras.__version__} is available")
    
    # Test if GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU support available")
    else:
        print("ℹ️  Running on CPU only")
        
except ImportError:
    print("❌ TensorFlow is not installed")
    print("Run: pip install tensorflow")