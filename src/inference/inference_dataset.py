import argparse
import pathlib
import random
import warnings
from typing import Literal

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from retinaface.pre_trained_models import get_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.from_hina_graduation.classifier.base import Classifier
from src.inference.datasets import (
    init_cdf,
    init_dfd,
    init_dfdc,
    init_dfdcp,
    init_ff,
    init_ffiw,
)
from src.inference.model import Detector
from src.inference.preprocess import extract_frames

warnings.filterwarnings("ignore")

device = torch.device("cuda")


def main(
    weight_path: pathlib.Path,
    model_name: Literal["efficientnet_b4"],
    dataset_name: Literal["FFIW", "FF", "DFD", "DFDC", "DFDCP", "CDF"],
    hina_graduation_config_path: pathlib.Path = pathlib.Path(
        "src/from_hina_graduation/config"
    ),
) -> None:
    """DeepFake検出器の性能を評価する. 性能はAUCの値で標準出力に出力される.

    Args:
        weight_path (pathlib.Path): 評価する重みが保存されているパス. tarか
            pthのみが許容される. ckptは事前にpth形式に変換する必要がある.
        model_name (Literal): 評価するDeepFake検出器のモデルの名前.
        dataset_name (Literal): 評価に使用するデータセットの名前.
        hina_graduation_config_path (pathlib.Path): hina_graduationから
            移動してきたconfigが配置されているディレクトリのパス.

    """
    # DeepFake検出器（分類器）の重みを読み込む
    # .tarで終わる場合はSelfBlendedImagesのコードで学習した重みを読み込む
    if weight_path.suffix == ".tar":
        model = Detector()
        model = model.to(device)
        cnn_sd = torch.load(weight_path)["model"]
        model.load_state_dict(cnn_sd)
    # .pthで終わる場合はClassifierクラスに重みを読み込む
    elif weight_path.suffix == ".pth":
        classifier_config_path = (
            hina_graduation_config_path / "classifier" / f"{model_name}.yaml"
        )
        classifier_config = OmegaConf.load(classifier_config_path)
        encoder_base = instantiate(classifier_config.encoder_base)

        classifier = Classifier(
            encoder_base=encoder_base,
            head_config=classifier_config.head,
            num_classes=2,
            use_encoder_base_head=False,
        )
        classifier.load_state_dict(torch.load(weight_path))
        model = classifier.to(device)
    else:
        raise NotImplementedError

    model.eval()

    # 顔検出器をインスタンス化.
    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    # データセットを初期化.
    if dataset_name == "FFIW":
        video_list, target_list = init_ffiw()
    elif dataset_name == "FF":
        video_list, target_list = init_ff()
    elif dataset_name == "DFD":
        video_list, target_list = init_dfd()
    elif dataset_name == "DFDC":
        video_list, target_list = init_dfdc()
    elif dataset_name == "DFDCP":
        video_list, target_list = init_dfdcp()
    elif dataset_name == "CDF":
        video_list, target_list = init_cdf()
    else:
        NotImplementedError(f"dataset `{dataset_name}` is not supported.")

    output_list = []
    for filename in tqdm(video_list):
        try:
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector)

            with torch.no_grad():
                img = torch.tensor(face_list).to(device).float() / 255
                if model.__class__.__name__ == "Detector":
                    # Detectorクラスではlogitsは[Real, Fake]の順です.
                    pred = model(img).softmax(1)[:, 1]
                elif model.__class__.__name__ == "Classifier":
                    # Classifierクラスではlogitsは[Fake, Real]の順です.
                    pred = model(img).logits.softmax(1)[:, 0]
                else:
                    raise NotImplementedError

            pred_list = []
            idx_img = -1
            for i in range(len(pred)):
                if idx_list[i] != idx_img:
                    pred_list.append([])
                    idx_img = idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res = np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i] = max(pred_list[i])
            pred = pred_res.mean()
        except Exception as e:
            print(e)
            pred = 0.5
        output_list.append(pred)

    auc = roc_auc_score(target_list, output_list)
    print(f"{dataset_name}| AUC: {auc:.4f}")


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", dest="weight_path", type=pathlib.Path, required=True)
    parser.add_argument(
        "-m",
        dest="model_name",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4"],
    )
    parser.add_argument(
        "-d",
        dest="dataset_name",
        type=str,
        choices=["FFIW", "FF", "DFD", "DFDC", "DFDCP", "CDF"],
    )
    parser.add_argument("-n", dest="n_frames", default=32, type=int)
    args = parser.parse_args()

    main(
        weight_path=args.weight_path,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
    )
