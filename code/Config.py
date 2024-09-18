from dataclasses import dataclass

@dataclass
class Config:
    host: str
    port: int
    username: str
    password: str
    timeout: int = 30  # Optional parameter with a default value
    use_ssl: bool = True  # Optional parameter with a default value