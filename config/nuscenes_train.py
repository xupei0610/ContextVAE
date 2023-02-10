MIN_OB_HORIZON = 2
OB_HORIZON = 5
PRED_HORIZON = 12
OB_RADIUS = 30
MAP_SIZE = 224


lr = 3e-4
epochs = 100
test_since = 60
preload_data = True
pred_samples = 5
clustering = 0

train_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["CHALLENGE"],
    batch_size=128,
    batches_per_epoch=200
)
test_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["CHALLENGE"],
    batch_size=512
)

model = dict(
    horizon = PRED_HORIZON,
    ob_radius = OB_RADIUS,
    hidden_dim = 512,
    map_model = "resnet18"
)
