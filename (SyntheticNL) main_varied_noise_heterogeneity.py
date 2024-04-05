from GSSFiltering.model import SyntheticNLModel
from GSSFiltering.filtering import Extended_Kalman_Filter
from GSSFiltering.filtering import (
    KalmanNet_Filter,
    Split_KalmanNet_Filter,
    KalmanNet_Filter_v2,
)
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester
import numpy as np
import configparser
import os

if not os.path.exists("./.results"):
    os.mkdir("./.results")

config = configparser.ConfigParser()
config.read("./config.ini")

TRAIN = True
# TRAIN = False

# Set different vdB to test effect of increasing noise heterogeneity
vdB_list = np.arange(0, 60 + 1, 10)  # 0 -> 60

train_iter = int(config["Train"]["train_iter"])

# S_KalmanNet
test_list = [str(train_iter), "best"]

test_loss_list_Kalman = []
test_loss_list_Kalman_v2 = []
test_loss_list_Split = []
test_loss_ekf = []
test_loss_ekf_mm = []

if TRAIN:
    for vdB in vdB_list:
        print(f"\nTraining with vdB = {vdB}")
        # Define model
        synthetic_nl_model = SyntheticNLModel(mode="train")
        synthetic_nl_model.v_dB = vdB
        synthetic_nl_model.init_params()
        synthetic_nl_model.save_path = f"./.data/syntheticNL_vdB_{vdB}/"
        TRAIN_DATA_PATH = f"./.data/syntheticNL_vdB_{vdB}/train/"
        os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
        synthetic_nl_model.generate_data()

        valid_synthetic_nl_model = SyntheticNLModel(mode="valid")
        valid_synthetic_nl_model.v_dB = vdB
        valid_synthetic_nl_model.init_params()
        valid_synthetic_nl_model.save_path = f"./.data/syntheticNL_vdB_{vdB}/"
        VALID_DATA_PATH = f"./.data/syntheticNL_vdB_{vdB}/valid/"
        os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
        valid_synthetic_nl_model.generate_data()
        # KalmanNet
        trainer_kalman = Trainer(
            dnn=KalmanNet_Filter(synthetic_nl_model),
            data_path=TRAIN_DATA_PATH,
            save_path=f"(syntheticNL) KalmanNet_vdB_{vdB}.pt",
            mode=0,
        )

        # KalmanNet (architecture 2)
        trainer_kalman_v2 = Trainer(
            dnn=KalmanNet_Filter_v2(synthetic_nl_model),
            data_path=TRAIN_DATA_PATH,
            save_path=f"(syntheticNL, v2) KalmanNet_vdB_{vdB}.pt",
            mode=0,
        )

        # S_KalmanNet
        trainer_split = Trainer(
            dnn=Split_KalmanNet_Filter(synthetic_nl_model),
            data_path="./.data/syntheticNL/train/",
            save_path=f"(syntheticNL) Split_KalmanNet_vdB_{vdB}.pt",
            mode=1,
        )
        for i in range(train_iter):
            trainer_split.train_batch()
            trainer_split.dnn.reset(clean_history=True)
            if trainer_split.train_count % trainer_split.save_num == 0:
                trainer_split.validate(
                    Tester(
                        filter=Split_KalmanNet_Filter(valid_synthetic_nl_model),
                        data_path=VALID_DATA_PATH,
                        model_path=f"./.model_saved/(syntheticNL) Split_KalmanNet_vdB_{vdB}_"
                        + str(trainer_split.train_count)
                        + ".pt",
                        is_validation=True,
                    )
                )

            trainer_kalman.train_batch()
            trainer_kalman.dnn.reset(clean_history=True)
            if trainer_kalman.train_count % trainer_kalman.save_num == 0:
                trainer_kalman.validate(
                    Tester(
                        filter=KalmanNet_Filter(valid_synthetic_nl_model),
                        data_path=VALID_DATA_PATH,
                        model_path=f"./.model_saved/(syntheticNL) KalmanNet_vdB_{vdB}_"
                        + str(trainer_kalman.train_count)
                        + ".pt",
                        is_validation=True,
                    )
                )

            trainer_kalman_v2.train_batch()
            trainer_kalman_v2.dnn.reset(clean_history=True)
            if trainer_kalman_v2.train_count % trainer_kalman_v2.save_num == 0:
                trainer_kalman_v2.validate(
                    Tester(
                        filter=KalmanNet_Filter_v2(valid_synthetic_nl_model),
                        data_path=VALID_DATA_PATH,
                        model_path=f"./.model_saved/(syntheticNL, v2) KalmanNet_vdB_{vdB}_"
                        + str(trainer_kalman_v2.train_count)
                        + ".pt",
                        is_validation=True,
                    )
                )

for vdB in vdB_list:
    print(f"\nTesting with vdB = {vdB}")
    test_syntheticnl = SyntheticNLModel(mode="test")
    test_syntheticnl.v_dB = vdB
    test_syntheticnl.init_params()
    test_syntheticnl.save_path = f"./.data/syntheticNL_vdB_{vdB}/"
    TEST_DATA_PATH = f"./.data/syntheticNL_vdB_{vdB}/test/"
    os.makedirs(TEST_DATA_PATH, exist_ok=True)
    test_syntheticnl.generate_data()

    test_syntheticnl_mm = SyntheticNLModel(mode="test")
    test_syntheticnl_mm.v_dB = 30
    test_syntheticnl_mm.q2_dB = -20
    test_syntheticnl_mm.init_params()

    # EKF perfect
    ekf_filter = Extended_Kalman_Filter(test_syntheticnl)
    tester_ekf = Tester(
        filter=ekf_filter,
        data_path=TEST_DATA_PATH,
        model_path="EKF ",
    )
    test_loss_ekf += [tester_ekf.loss.item()]

    # EKF mismatch
    ekf_mm_filter = Extended_Kalman_Filter(test_syntheticnl_mm)
    tester_ekf_mm = Tester(
        filter=ekf_mm_filter,
        data_path=TEST_DATA_PATH,
        model_path="EKF_mm ",
    )
    test_loss_ekf_mm += [tester_ekf_mm.loss.item()]

    # KalmanNetv1
    tester_kf = Tester(
        filter=KalmanNet_Filter(test_syntheticnl),
        data_path=TEST_DATA_PATH,
        model_path=f"./.model_saved/(syntheticNL) KalmanNet_vdB_{vdB}_best.pt",
    )
    test_loss_list_Kalman += [tester_kf.loss.item()]

    # KalmanNetv2
    tester_kf2 = Tester(
        filter=KalmanNet_Filter_v2(test_syntheticnl),
        data_path=TEST_DATA_PATH,
        model_path=f"./.model_saved/(syntheticNL, v2) KalmanNet_vdB_{vdB}_best.pt",
    )
    test_loss_list_Kalman_v2 += [tester_kf2.loss.item()]

    # Split-KalmanNet
    tester_skf = Tester(
        filter=Split_KalmanNet_Filter(test_syntheticnl),
        data_path=TEST_DATA_PATH,
        model_path=f"./.model_saved/(syntheticNL) Split_KalmanNet_vdB_{vdB}_best.pt",
    )
    test_loss_list_Split += [tester_skf.loss.item()]

    print(f"test_EKF (perfect) Loss{test_loss_ekf}")
    print(f"test_EKF (mismatch) Loss{test_loss_ekf_mm}")
    print(f"test_KalmanNetv1 Loss{test_loss_list_Kalman}")
    print(f"KalmanNetv2 Loss{test_loss_list_Kalman_v2}")
    print(f"Split-KalmanNet Loss{test_loss_list_Split}")

    np.save("./.results/test_loss_ekf_noise_hetero.npy", np.array(test_loss_ekf))
    np.save("./.results/test_loss_ekf_mm_noise_hetero.npy", np.array(test_loss_ekf_mm))
    np.save("./.results/test_loss_kalman_noise_hetero.npy", np.array(test_loss_list_Kalman))
    np.save(
        "./.results/test_loss_kalman_v2_noise_hetero.npy",
        np.array(test_loss_list_Kalman_v2),
    )
    np.save("./.results/test_loss_split_noise_hetero.npy", np.array(test_loss_list_Split))
