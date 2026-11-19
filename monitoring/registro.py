
import logging
import time

from fastapi import Request


# Configurar los registros de monitoreo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("monitoreo_sentimientos")


def configurar_monitoreo(app):
    """Registrar solicitudes, estado y tiempo de respuesta."""

    @app.middleware("http")
    async def registrar_solicitud(request: Request, call_next):
        inicio = time.perf_counter()

        try:
            response = await call_next(request)

            duracion_ms = (
                time.perf_counter() - inicio
            ) * 1000

            logger.info(
                "ruta=%s | metodo=%s | estado=%s | duracion_ms=%.2f",
                request.url.path,
                request.method,
                response.status_code,
                duracion_ms,
            )

            return response

        except Exception:
            duracion_ms = (
                time.perf_counter() - inicio
            ) * 1000

            logger.exception(
                "Error en solicitud | ruta=%s | duracion_ms=%.2f",
                request.url.path,
                duracion_ms,
            )

            raise
