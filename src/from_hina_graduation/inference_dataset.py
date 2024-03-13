import argparse
import json
import pathlib
import random
import warnings
from typing import List, Literal, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from retinaface.pre_trained_models import get_model
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm

from src.from_hina_graduation.classifier.base import Classifier
from src.inference.model import Detector

warnings.filterwarnings("ignore")

device = torch.device("cuda")


class InferenceDataset(Dataset):
    """SBIsに実装されている評価コードで使用するデータを扱うデータセット."""

    def __init__(
        self,
        dataset_root_path: pathlib.Path,
    ) -> None:
        """SBIsに実装されている評価コードで使用するデータを扱うデータセット. この
        クラスを使用するためには事前に export_dataset_for_inference.py を使用
        してデータを出力しておく必要があります.

        Args:
            dataset_root_path (pathlib.Path): inference用のデータが配置され
                ているディレクトリのルートパス. 例えば, FaceForensics++では
                "./data/FaceForensics++/inference"などになる.

        """
        super().__init__()
        # target_listをJSONファイルから読み込む.
        target_list_path = dataset_root_path / "target_list.json"
        with target_list_path.open("r") as f:
            self.target_list = json.load(f)

        # no_face.txtをListに格納
        self.no_face_paths: List[pathlib.Path] = list(
            dataset_root_path.glob("*/no_face.txt")
        )

        # dataset_root_path直下のディレクトリをすべて取得
        all_dir_paths: List[pathlib.Path] = list(dataset_root_path.glob("*/"))
        face_dir_paths: List[pathlib.Path] = list(
            set(all_dir_paths) - set(path.parent for path in self.no_face_paths)
        )

        # 画像ファイルをListに格納
        self.face_paths: List[List[pathlib.Path]] = [
            list(path.glob("*.png")) for path in face_dir_paths
        ]

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[int]]:
        """データセットからデータを取得する."""
        face_list: List[np.ndarray] = []
        idx_list: List[int] = []

        for image_path in self.face_paths[index]:
            # 画像ファイルをnp.ndarrayに読み込んで, face_listに格納する.
            image = Image.open(image_path).convert("RGB")
            face_list.append(np.array(image).transpose(2, 0, 1))

            # 画像ファイルは`<通し番号>_<frame番号>.png`の形で保存されているので
            # frame番号の部分を抜き出して, idx_listに格納する.
            idx_list.append(image_path.stem.split("_")[-1])

        return face_list, idx_list

    def __len__(self) -> int:
        """データセットの長さを返却する."""
        return len(self.face_paths)

    @property
    def num_no_face(self) -> int:
        """顔が検出されなかった動画の数を返却する."""
        return len(self.no_face_paths)


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
        raise NotImplementedError
    elif dataset_name == "FF":
        dataset_root_path = pathlib.Path("data/FaceForensics++/inference")
        inference_dataset = InferenceDataset(dataset_root_path)
    elif dataset_name == "DFD":
        raise NotImplementedError
    elif dataset_name == "DFDC":
        raise NotImplementedError
    elif dataset_name == "DFDCP":
        raise NotImplementedError
    elif dataset_name == "CDF":
        raise NotImplementedError
    else:
        NotImplementedError(f"dataset `{dataset_name}` is not supported.")

    output_list = []
    for face_list, idx_list in tqdm(inference_dataset, total=len(inference_dataset)):
        try:
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
            pred = 0.5

        output_list.append(pred)

    # 顔が検出されなかった動画については0.5を追加
    output_list.extend([0.5 for _ in range(inference_dataset.num_no_face)])

    auc = roc_auc_score(inference_dataset.target_list, output_list)
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
