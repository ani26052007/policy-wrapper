MODALITY_CONFIGS: dict = {}


def register_modality_config(config, embodiment_tag):
    assert embodiment_tag.value not in MODALITY_CONFIGS, f"{embodiment_tag} already registered"
    MODALITY_CONFIGS[embodiment_tag.value] = config
