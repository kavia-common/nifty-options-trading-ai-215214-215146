import json
import os

from src.api.main import app

# PUBLIC_INTERFACE
def generate_openapi(output_dir: str = "interfaces", filename: str = "openapi.json") -> str:
    """Generate and write the OpenAPI schema to interfaces/openapi.json"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    schema = app.openapi()
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    return output_path


if __name__ == "__main__":
    path = generate_openapi()
    print(f"Wrote OpenAPI to {path}")
