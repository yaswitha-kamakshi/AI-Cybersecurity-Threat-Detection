import numpy as np
from src.simulator import generate_traffic
from src.preprocessing import clean, engineer_features, prepare_datasets


def make_df():
    return generate_traffic(n=4000, seed=1)


def test_clean_keeps_valid_rows():
    df, report = clean(make_df())
    assert report["rows_out"] == len(df) > 0


def test_features_have_no_missing_or_infinite_values():
    X = engineer_features(make_df())
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()


def test_split_is_stratified_and_disjoint():
    data = prepare_datasets(make_df())
    assert set(data["X_train"].index).isdisjoint(data["X_test"].index)
    train_share = (data["y_train"] == "DoS").mean()
    test_share = (data["y_test"] == "DoS").mean()
    assert abs(train_share - test_share) < 0.01


def test_scaler_uses_training_data_only():
    data = prepare_datasets(make_df())
    assert abs(data["X_train_scaled"].mean().abs().max()) < 1e-6      # train is centred
    assert data["X_test_scaled"].mean().abs().max() > 1e-6            # test is not forced to be