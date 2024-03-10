import argparse
import pathlib
import random
import warnings
from typing import Literal

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from model import Detector
from omegaconf import OmegaConf
from preprocess import extract_face
from retinaface.pre_trained_models import get_model

from src.from_hina_graduation.classifier.base import Classifier

warnings.filterwarnings("ignore")

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")


def main(
    weight_path: pathlib.Path,
    image_path: pathlib.Path,
    model_name: Literal["efficientnet_b4"],
    hina_graduation_config_path: pathlib.Path = pathlib.Path(
        "src/from_hina_graduation/config"
    ),
) -> None:
    """DeepFake検出器を使用して画像のFakenessを評価する. Fakenessは標準出力に
    出力される.

    Args:
        weight_path (pathlib.Path): 評価する重みが保存されているパス. tarか
            pthのみが許容される. ckptは事前にpth形式に変換する必要がある.
        image_path (pathlib.Path): Fakenessを評価する画像のパス.
        model_name (Literal): 評価するDeepFake検出器のモデルの名前.
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

    frame = cv2.imread(str(image_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔検出器をインスタンス化.
    face_detector = get_model(
        "resnet50_2020-07-20", max_size=max(frame.shape), device=device
    )
    face_detector.eval()

    face_list = extract_face(frame, face_detector)

    with torch.no_grad():
        img = torch.tensor(face_list).to(device).float() / 255
        if model.__class__.__name__ == "Detector":
            # Detectorクラスではlogitsは[Real, Fake]の順です.
            pred = model(img).softmax(1)[:, 1].cpu().data.numpy().tolist()
        elif model.__class__.__name__ == "Classifier":
            # Classifierクラスではlogitsは[Fake, Real]の順です.
            pred = model(img).logits.softmax(1)[:, 0].cpu().data.numpy().tolist()
        else:
            raise NotImplementedError

    print(f"fakeness: {max(pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", dest="weight_path", type=pathlib.Path, required=True)
    parser.add_argument("-i", dest="image_path", type=pathlib.Path, required=True)
    parser.add_argument(
        "-m",
        dest="model_name",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4"],
    )
    args = parser.parse_args()

    main(
        weight_path=args.weight_path,
        image_path=args.image_path,
        model_name=args.model_name,
    )
