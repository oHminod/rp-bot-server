"""Contrat de découverte stable du serveur HTTP PuLID."""

from __future__ import annotations


API_CONTRACT_VERSION = "1.2.0"


def capabilities_payload(
    *,
    application_version: str,
    embedding_model: str | None,
    embedding_dimensions: int | None,
) -> dict[str, object]:
    """Décrit les fonctions HTTP sans inspecter ni charger leurs modèles."""

    return {
        "component": "pulid",
        "version": application_version,
        "api_contract_version": API_CONTRACT_VERSION,
        "capabilities": {
            "krea2_generation": {
                "enabled": True,
                "generation_endpoint": "/generate/krea2",
                "catalog_endpoint": "/models/krea2",
                "identity_transfer": False,
                "samplers": ["euler"],
                "schedulers": ["beta"],
            },
            "image_generation": {
                "enabled": True,
                "catalog_endpoint": "/models",
                "generation_endpoint": "/generate",
            },
            "text_embeddings": {
                "enabled": embedding_model is not None,
                "models_endpoint": "/v1/models",
                "embeddings_endpoint": "/v1/embeddings",
                "model": embedding_model,
                "dimensions": embedding_dimensions,
            },
        },
    }


__all__ = ["API_CONTRACT_VERSION", "capabilities_payload"]
