OB_HORIZON = 6
PRED_HORIZON = 15
OB_RADIUS = 30
MAP_SIZE = 224


lr = 3e-4
epochs = 400
test_since = 200
preload_data = False
pred_samples = 5
clustering = 0

train_dataloader = dict(
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["EGO", "VEHICLE"],
    batch_size=512,
    batches_per_epoch=500
)
test_dataloader = dict(
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["EGO", "VEHICLE"],
    # let each sub-trajectory has at most 7 frame overlapped
    # this setting is to reduce the data size when testing during training
    traj_max_overlap=7, 
    batch_size=256
)

model = dict(
    horizon = PRED_HORIZON,
    ob_radius = OB_RADIUS,
    hidden_dim = 512,
    map_model = "resnet152"
)
