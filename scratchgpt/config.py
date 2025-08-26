from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ScratchGPTArchitecture(BaseSettings):
    """
    All settings for training the model.
    """

    block_size: int = 256
    embedding_size: int = 384
    """ Size of the individual embeddings vector """
    num_heads: int = 6
    num_blocks: int = 6
    vocab_size: int | None = None

    model_config = SettingsConfigDict(
        env_prefix="ARCHITECTURE_",
        extra="allow",
    )


class ScratchGPTTraining(BaseSettings):
    """
    All training related parameters
    """

    max_epochs: int = 50
    learning_rate: float = 3e-4
    batch_size: int = 32
    dropout_rate: float = 0.2
    random_seed: int = 1337

    model_config = SettingsConfigDict(
        env_prefix="TRAINING_",
        extra="allow",
    )


class ScratchGPTConfig(BaseSettings):
    """
    Full model config
    """

    architecture: ScratchGPTArchitecture = Field(default_factory=ScratchGPTArchitecture)
    training: ScratchGPTTraining = Field(default_factory=ScratchGPTTraining)

    model_config = SettingsConfigDict(
        env_prefix="SCRATCH_GPT_",
        extra="allow",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file="scratch_gpt.yaml"),
        )
