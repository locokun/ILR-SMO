class Config:
    # General
    SEED = 42
    DEVICE = "cuda"
    
    # Model
    EMB_DIM = 256
    HIDDEN_DIMS_MAX_LAYERS = 2
    
    # Training
    PRETRAIN_EPOCHS = 500
    PRETRAIN_BATCH_SIZE = 4096
    PRETRAIN_LR = 1e-3
    PRETRAIN_WEIGHT_DECAY = 1e-4
    
    TRAIN_EPOCHS = 350
    TRAIN_BATCH_SIZE = 1024
    TRAIN_LR_AE = 5e-5
    TRAIN_LR_FUSION = 1e-3
    TRAIN_WEIGHT_DECAY = 1e-4
    NOISE_SIGMA = 0.02
    
    # Loss Coefficients
    COEF_REC = 1.0
    COEF_CONTRASTIVE = 1.0
    COEF_ALIGN = 1.0
    COEF_SPATIAL = 1.0
    COEF_VAR = 1.0
    COEF_ENTROPY = 0.1
    
    # Evaluation
    EVAL_INTERVAL = 5
    TARGET_K = 10
    CLUSTERING_ALGO = "leiden"
    SPATIAL_K = 6
    
    # Paths
    SAVE_DIR = "."
    BEST_MODEL_NAME = "best_model_ari.pth"

    @classmethod
    def set_dataset(cls, dataset_name):
        if dataset_name == "MouseBrain":
            # Restore defaults for MouseBrain
            cls.TARGET_K = 10
            cls.TRAIN_EPOCHS = 500
            cls.CLUSTERING_ALGO = "leiden"
            cls.TRAIN_LR_FUSION = 1e-3
            cls.COEF_REC = 1.0
            cls.COEF_CONTRASTIVE = 1.0
            cls.COEF_SPATIAL = 1.0 # 1.0
            cls.COEF_ENTROPY = 0.1
            
        elif dataset_name == "HumanLymphNodeNoImage":
            # Use HumanLymphNode config as base, but tuned for NoImage case
            cls.PRETRAIN_EPOCHS = 500
            cls.TRAIN_EPOCHS = 100
            cls.TARGET_K = 7
            cls.CLUSTERING_ALGO = "leiden"
            
            # Learning Rates
            cls.TRAIN_LR_FUSION = 5e-4
            
            # Loss Coefficients (Adopting HumanLymphNode values to boost Stage 2 performance)
            cls.COEF_REC = 0.5
            cls.COEF_CONTRASTIVE = 8.0  # High contrastive to improve ARI
            cls.COEF_SPATIAL = 1.0      # Maintain spatial coherence
            cls.COEF_ALIGN = 0.3
            cls.COEF_ENTROPY = 0.5
            cls.COEF_VAR = 8.0          # High variance to prevent collapse
            
        elif dataset_name == "HumanHippocampus":
            # Best Performance Configuration (ARI ~0.5340)
            cls.EMB_DIM = 256
            cls.HIDDEN_DIMS_MAX_LAYERS = 2
            cls.TARGET_K = 7
            cls.CLUSTERING_ALGO = "leiden"
            cls.PRETRAIN_EPOCHS = 600
            cls.TRAIN_EPOCHS = 350
            cls.TRAIN_LR_FUSION = 8e-4
            cls.COEF_REC = 0.5
            cls.COEF_CONTRASTIVE = 12.0
            cls.COEF_SPATIAL = 3.0
            cls.COEF_ALIGN = 2.0
            cls.COEF_ENTROPY = 1.0
            cls.COEF_VAR = 10.0
            
        elif dataset_name == "TripletOmics":
            cls.TARGET_K = 5
            cls.CLUSTERING_ALGO = "leiden"
            cls.TRAIN_EPOCHS = 200
            cls.TRAIN_LR_FUSION = 1e-3
            cls.COEF_REC = 0.5
            cls.COEF_CONTRASTIVE = 2.0
            cls.COEF_SPATIAL = 2.0
            cls.COEF_ALIGN = 1.0
            cls.COEF_ENTROPY = 1.0
            cls.COEF_VAR = 2.0
        
        elif dataset_name == "MVC_simulated":
            cls.PRETRAIN_EPOCHS = 800
            cls.TRAIN_EPOCHS = 300
            cls.TARGET_K = 6
            cls.CLUSTERING_ALGO = "kmeans"
            cls.TRAIN_LR_FUSION = 5e-4
            cls.COEF_SPATIAL = 10.0
            cls.COEF_ALIGN = 10.0
            cls.COEF_REC = 0.5
            cls.COEF_CONTRASTIVE = 8.0
            cls.COEF_SPATIAL = 1.0 # 1.0  # Increased spatial regularization
            cls.COEF_ALIGN = 0.3
            cls.COEF_ENTROPY = 0.5
            cls.COEF_VAR = 8.0
