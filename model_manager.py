from pathlib import Path
from urllib.request import urlopen


def ensure_model(
    model_path: Path,
    model_url: str,
) -> Path:
    """
    Baixa o modelo Face Landmarker apenas quando ele ainda não existe.
    """
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    print("Baixando o modelo facial do MediaPipe...")
    print(model_url)

    try:
        with urlopen(model_url, timeout=90) as response:
            model_data = response.read()
    except Exception as error:
        raise RuntimeError(
            "Nao foi possivel baixar o modelo Face Landmarker. "
            "Verifique sua conexao com a internet e tente novamente."
        ) from error

    if not model_data:
        raise RuntimeError(
            "O download do modelo Face Landmarker retornou um arquivo vazio."
        )

    model_path.write_bytes(model_data)

    print(
        "Modelo salvo em:",
        model_path,
    )

    return model_path