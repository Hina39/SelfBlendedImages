import copy
from collections import OrderedDict
from typing import Final, Iterable, NamedTuple, Optional, cast

import hydra
import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig


class ClassifierForwardReturn(NamedTuple):
    """Classifierクラスのforwardメソッドからの返り値として使うNamedTuple.

    Attributes:
        features (torch.Tensor): encoder(特徴量抽出器)からの出力.
        logits (torch.Tensor): head(分類器)からの出力.

    """

    features: torch.Tensor
    logits: torch.Tensor


class Classifier(nn.Module):
    """クラス分類器として動作するtorch.nn.module.

    encoder(特徴量抽出器)とhead(分類器)を属性として保持し, クラス分類器として動作
    するクラス. forwardの際はfeaturesとlogitsの両方を返却する.

    x -> (encoder) -> features -> (head) -> logits

    Attributes:
        encoder (torch.nn.Module): encoder(特徴量抽出器).
        head_config (DictConfig): encoder_baseの中で, headの情報を持っている
            DictConfig. 最後の層はtorch.nn.Linearであることを想定.

    Note:
        head_configは以下のような構成を前提としている.

        {
            'fc1': [
                {
                    '_target_': 'torch.nn.Linear',
                    'in_features': 2048,
                    'out_features': 1024,
                    'bias': True
                }
            ],
            'fc2': [
                {
                    '_target_': 'torch.nn.Linear',
                    'in_features': 1024,
                    'out_features': 10,
                    'bias': True
                }
            ],
        }

    """

    def __init__(
        self,
        encoder_base: nn.Module,
        head_config: DictConfig,
        num_classes: Optional[int],
        use_encoder_base_head: bool = False,
    ) -> None:
        """Classifierクラスを初期化.

        Args:
            encoder_base (torch.nn.Module): encoder(特徴量抽出器)のベースと
                なるクラス分類器. encoderとheadから構成されていることを想定.
            head_config (DictConfig): encoder_baseの中で, headの情報を持って
                いるDictConfig. 最後の層はtorch.nn.Linearであることを想定.
            num_classes (Optional[int]): 分類タスクのクラス数. head(分類器)の最終層の
                torch.nn.Linearのout_featuresを決める引数.
            use_encoder_base_head (bool): base_encoderのheadの重みを引継ぐ.

        """
        super().__init__()
        self.head_config: Final = head_config

        # use_encoder_base_headとnum_classesは片方のみが指定されるべきなので,
        # XORをとる.
        assert (use_encoder_base_head is False) ^ (num_classes is None)

        # inplace処理を行うので, encoder_baseのコピーを作成する.
        _encoder_base = copy.deepcopy(encoder_base)

        self.head = self.create_new_head(self.head_config, num_classes)
        if use_encoder_base_head:
            for key in self.head_config:
                setattr(
                    self.head, cast(str, key), getattr(_encoder_base, cast(str, key))
                )

        self.encoder: Final = self._init_encoder(
            _encoder_base, [cast(str, k) for k in self.head_config.keys()]
        )

    def forward(self, x: torch.Tensor) -> ClassifierForwardReturn:
        """順伝搬を行う.

        Args:
            x (torch.Tensor): 入力データ.

        Returns:
            ClassifierForwardReturn: 順伝搬の結果. featuresとlogitsを含む.

        """
        features: Final = self.forward_features(x)
        logits: Final = self.forward_head(features)
        return ClassifierForwardReturn(
            features=features,
            logits=logits,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """encoder(特徴量抽出器)のみ順伝播を行う.

        Args:
            x (torch.Tensor): 入力データ.

        Returns:
            torch.Tensor: features(特徴量)の情報を持つテンソル.

        """
        return self.encoder(x)  # type: ignore

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        """head(分類器)のみ順伝播を行う.

        Args:
            features (torch.Tensor): features(特徴量)の情報を持つテンソル.

        Returns:
            torch.Tensor: logitsの情報を持つテンソル.

        """
        return self.head(features)  # type: ignore

    def _init_encoder(
        self,
        encoder_base: nn.Module,
        deactivate_targets: Iterable[str],
    ) -> nn.Module:
        """encoder_baseのモジュールのうちdeactivate_targetsに含まれるものを
        非有効化する(恒等写像に置き換える). このメソッドはencoder_baseからhead
        (分類器)の役割を果たすモジュールを取り除くのに使用される.

        Args:
            encoder_base (torch.nn.Module): encoder(特徴量抽出器)のベースと
                なるクラス分類器. encoderとheadから構成されていることを想定.
            deactivate_targets (Iterable[str]): encoder_baseの中で非有効化
                するモジュールの名前.

        Returns:
            torch.nn.Module: deactivate_targetsが恒等写像に置き換えられた
                encoder_base.

        """
        for target in deactivate_targets:
            setattr(encoder_base, target, nn.Identity())

        return encoder_base

    def create_new_head(
        self,
        head_config: DictConfig,
        num_classes: Optional[int],
    ) -> nn.Module:
        """新しいhead(分類器)のオブジェクトを作成する. head_configに記載されて
        いる順番でモジュールをインスタンス化し, torch.nn.Sequentialとして返却.

        Args:
            head_config (DictConfig): encoder_baseの中で, headの情報を持って
                いるDictConfig. 最後の層はtorch.nn.Linearであることを想定.
            num_classes (Optional[int]): 分類タスクのクラス数. head(分類器)の
                最終層のtorch.nn.Linearのout_featuresを決める引数.

        Returns:
            torch.nn.Module: 新しいhead(分類器)オブジェクト.

        """
        module_dict: Final[OrderedDict[str, nn.Module]] = OrderedDict()
        for i, (k, v) in enumerate(head_config.items()):
            # num_classesがNoneでないときは, 最後のモジュールのout_featuresを
            # 変更する.
            if (i == len(head_config.items()) - 1) and (num_classes is not None):
                # 最後のモジュールはtorch.nn.Linearであることを想定しています.
                assert v[0]["_target_"] == "torch.nn.Linear"
                module_dict[cast(str, k)] = hydra.utils.instantiate(
                    *v, out_features=num_classes
                )
                continue

            module_dict[cast(str, k)] = hydra.utils.instantiate(*v)
        return nn.Sequential(module_dict)

    def freeze_encoder(self) -> None:
        """encoder(特徴量抽出器)のパラメータを固定する."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """encoder(特徴量抽出器)のパラメータを解放する."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    @property
    def num_classes(self) -> int:
        """分類タスクのクラス数を返却する.

        Returns:
            int: 分類タスクのクラス数.

        """
        return cast(int, cast(nn.Sequential, self.head)[-1].out_features)


if __name__ == "__main__":
    import torchvision
    from omegaconf import OmegaConf

    # resnet50をencoder_baseとしてClassifierクラスを使用する際の例.
    x = torch.randn(16, 3, 32, 32)
    encoder_base = torchvision.models.resnet50()
    print(f"forward result shape of encoder_base: {encoder_base(x).shape}")

    head_config = OmegaConf.create(
        {
            "fc": [
                {
                    "_target_": "torch.nn.Linear",
                    "in_features": 2048,
                    "out_features": 1024,
                    "bias": True,
                }
            ],
        }
    )

    classifier = Classifier(
        encoder_base=encoder_base,
        head_config=head_config,
        num_classes=10,
    )
    print(classifier)
    print("forward result shape of classifier:")
    print(f"- features: {classifier(x).features.shape}")
    print(f"- logits: {classifier(x).logits.shape}")
    print(f"- num_classes: {classifier.num_classes}")

    print("test freeze and unfreeze encoder")
    print([param.requires_grad for param in classifier.encoder.parameters()])
    classifier.freeze_encoder()
    print([param.requires_grad for param in classifier.encoder.parameters()])
    classifier.unfreeze_encoder()
    print([param.requires_grad for param in classifier.encoder.parameters()])
