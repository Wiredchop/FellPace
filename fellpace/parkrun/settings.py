from pydantic import BaseModel, field_validator, ValidationError, PositiveInt
import toml

class LocationSettings(BaseModel):
    start_ID: PositiveInt
    climb: PositiveInt


class PRSettings(BaseModel):
    hillsborough: LocationSettings
    endcliffe: LocationSettings

    @classmethod
    def load_toml_settings(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = toml.load(f)
        return cls(**data)

try:
    settings = PRSettings.load_toml_settings('settings.toml')
except ValidationError as e:
    print(e)
    settings = None
