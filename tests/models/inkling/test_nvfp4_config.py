from types import SimpleNamespace

from vllm.models.inkling.nvfp4 import InklingNvfp4Config


def test_exl3_quantization_is_not_nvfp4() -> None:
    config = SimpleNamespace(
        quantization_config={
            "quant_method": "exl3",
            "bits": 2.5,
            "codebook": "mcg",
        }
    )

    assert InklingNvfp4Config.from_hf_config(config) is None


def test_modelopt_nvfp4_quantization_is_detected() -> None:
    config = SimpleNamespace(
        quantization_config={
            "modelopt_quant_config": {
                "quant_cfg": {
                    "*weight_quantizer": {
                        "num_bits": [2, 1],
                        "block_sizes": {"scale_bits": [4, 3]},
                    }
                }
            },
            "group_size": 16,
            "exclude_modules": ["model.llm.layers.2.mlp.shared_experts"],
        }
    )

    result = InklingNvfp4Config.from_hf_config(config)

    assert result is not None
    assert result.group_size == 16
    assert result.experts_quantized(2)
    assert not result.shared_experts_quantized(2)
