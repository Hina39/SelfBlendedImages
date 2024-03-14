"""inferenceのたびに動画から画像を抜いてくるのは非効率なので, クロップされた画像を保存
しておくスクリプト.

出力は, `data/`以下に次のように保存される. 画像のファイル名は
`<通し番号>_<frame番号>.png`の形になっている. 同じフレーム内に顔が複数検出される
ことがあるので通し番号をつけている.
顔がすべてのフレームで検出されなかった場合はno_face.txtを保存する. 中身は空ファイル.

Celeb-DF-v2/
│
└── inference/
     │
     ├── Celeb-synthesis_videos_id30_id3_0002/
     │    │
     │    ├── 0_0.png
     │    │
     │    ├── 1_10.png
     ...
     └── target_list.json

FaceForensics++
│
└── inference/
     │
     ├── manipulated_sequences_NeuralTextures_c23_videos_995_233/
     │    │
     │    ├── 0_0.png
     │    │
     │    ├── 1_9.png
     │    ...
     │
     ├── manipulated_sequences_Deepfakes_c23_videos_024_073/
     │    │
     │    └── no_face.txt
     ...
     └── target_list.json

"""

import argparse
import pathlib
import random
import warnings
from typing import Literal

import numpy as np
import torch
from PIL import Image
from retinaface.pre_trained_models import get_model
from tqdm import tqdm

from src.inference.datasets import (
    init_cdf,
    init_dfd,
    init_dfdc,
    init_dfdcp,
    init_ff,
    init_ffiw,
)
from src.inference.preprocess import extract_frames

warnings.filterwarnings("ignore")

device = torch.device("cuda")


def main(
    dataset_name: Literal["FFIW", "FF", "DFD", "DFDC", "DFDCP", "CDF"],
    phase_name: str = "inference",
    n_frames: int = 32,
    force_generate: bool = False,
) -> None:
    """inferenceのたびに動画から画像を抜いてくるのは非効率なので, クロップされた
    画像を保存する.

    Args:
        dataset_name (Literal): 評価に使用するデータセットの名前.
        phase_name (str): 保存時のルートディレクトリ名にこの値が使用される.
        n_frames (int): 動画から画像を切り出すフレームの枚数.
        force_generate (bool): Trueならすべての処理を最初から行います. False
            なら既に存在している画像の保存はスキップします.

    """
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

    if len(video_list) == 0:
        raise ValueError("No video data found.")

    # 以下の処理は相対パスを前提にしているので絶対パスだった場合は例外をあげます.
    assert not pathlib.Path(video_list[0]).is_absolute()
    # `data/Celeb-DF-v2/Celeb-synthesis/videos/id30_id3_0002.mp4`
    dataset_root_path = pathlib.Path(*pathlib.Path(video_list[0]).parts[:2])

    error_video_paths = []
    for filename, target in tqdm(zip(video_list, target_list), total=len(video_list)):
        # filenameからディレクトリ名を構成する.
        # 例. `manipulated_sequences_Deepfakes_c23_videos_000_003`
        dir_name = (
            str(
                pathlib.Path(filename).relative_to(dataset_root_path).with_suffix("")
            ).replace("/", "_")
            + f"_{target}"
        )

        try:
            # 顔の検出を行う.
            face_list, idx_list = extract_frames(filename, n_frames, face_detector)

            # 検出された顔ごとにフレームを保存する.
            for i, (face, idx) in enumerate(zip(face_list, idx_list)):
                save_path = dataset_root_path / phase_name / dir_name / f"{i}_{idx}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # 既に存在している場合, force_generateがFalseなら保存をスキップ.
                if save_path.exists() and (force_generate is False):
                    print(f"`{save_path}` already exist and skip it.")
                    continue

                Image.fromarray(face.transpose(1, 2, 0), "RGB").save(save_path)

                # 読み出して同じになるか確認
                loaded_image = Image.open(save_path).convert("RGB")
                loaded_image_np = np.array(loaded_image).transpose(2, 0, 1)
                assert np.allclose(face, loaded_image_np)

        # 動画内で顔が検出されなかった場合.
        except Exception as e:
            print(e)
            save_path = dataset_root_path / phase_name / dir_name / "no_face.txt"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_path.touch(exist_ok=True)
            error_video_paths.append(filename)

    # chmodで権限を再帰的に変更する.
    phase_dir_path = dataset_root_path / phase_name
    phase_dir_path.chmod(0o777)
    for child in phase_dir_path.rglob("*"):
        child.chmod(0o777)

    print(f"{len(video_list)} videos are proecessed.")
    print(f"{len(error_video_paths)} videos are failed to detect face.")
    if len(error_video_paths):
        print(error_video_paths)


if __name__ == "__main__":
    # シードを固定.
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        dest="dataset_name",
        type=str,
        choices=["FFIW", "FF", "DFD", "DFDC", "DFDCP", "CDF"],
    )
    args = parser.parse_args()

    main(dataset_name=args.dataset_name)
