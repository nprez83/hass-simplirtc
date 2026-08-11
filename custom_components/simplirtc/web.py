"""Web views for the SimpliRTC integration."""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from homeassistant.components.camera import DATA_COMPONENT
from homeassistant.components.ffmpeg import get_ffmpeg_manager
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

from .camera import SimpliSafeGo2rtcCamera, SimpliSafeLiveKitCamera
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

STREAM_WIDTH = 640


class SimpliRTCStreamInfoView(HomeAssistantView):
    """Handle SimpliRTC LiveKit stream-info requests."""

    url = "/api/simplirtc_proxy/{entity_id}"
    name = f"api:{DOMAIN}:simplirtc"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    async def get(self, request: web.Request, entity_id: str) -> web.Response:
        component = self.hass.data[DATA_COMPONENT]
        camera = component.get_entity(entity_id)

        if not isinstance(camera, SimpliSafeLiveKitCamera):
            return web.Response(
                status=404,
                text=f"Entity {entity_id} is not a SimpliSafeLiveKitCamera",
            )

        try:
            url, token = await camera._live_view()
            return web.json_response({"url": url, "token": token})
        except Exception as err:
            _LOGGER.exception(
                "Error fetching LiveKit stream info for %s",
                entity_id,
            )
            return web.Response(
                status=500,
                text=f"Error fetching stream info: {err}",
            )


class SimpliRTCFlvProxyView(HomeAssistantView):
    """Proxy SimpliSafe FLV with corrected audio timestamp repair.

    Narrow test:
      * H.264 video is stream-copied unchanged.
      * AAC audio is decoded, timestamp-repaired with aresample, then
        re-encoded to AAC.
      * Output remains FLV for go2rtc native HTTP/FLV ingest.
      * No automatic reconnect loop is added.

    This is intended to approximate the SimpliSafe web player's behavior
    when it encounters large audio timestamp gaps: preserve the video while
    repairing the audio timeline by filling/trimming audio as needed.
    """

    url = "/api/simplirtc_flv/{entity_id}"
    name = f"api:{DOMAIN}:flv"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    async def get(
        self,
        request: web.Request,
        entity_id: str,
    ) -> web.StreamResponse:
        component = self.hass.data[DATA_COMPONENT]
        camera = component.get_entity(entity_id)

        if not isinstance(camera, SimpliSafeGo2rtcCamera):
            return web.Response(
                status=404,
                text=f"Entity {entity_id} is not a SimpliRTC FLV camera",
            )

        if request.query.get("sig") != camera.proxy_token:
            _LOGGER.warning(
                "Invalid SimpliRTC FLV proxy signature for %s",
                entity_id,
            )
            return web.Response(status=401, text="Invalid signature")

        token = camera.access_token
        if not token:
            _LOGGER.error(
                "No SimpliSafe access token available for %s",
                entity_id,
            )
            return web.Response(
                status=503,
                text="No SimpliSafe access token available",
            )

        try:
            binary = get_ffmpeg_manager(self.hass).binary
        except Exception:
            _LOGGER.debug(
                "Could not obtain Home Assistant FFmpeg binary; "
                "falling back to ffmpeg in PATH"
            )
            binary = "ffmpeg"

        flv_url = camera.flv_url(width=STREAM_WIDTH)

        headers = (
            f"Authorization: Bearer {token}\r\n"
            "Accept: */*\r\n"
            "Origin: https://webapp.simplisafe.com\r\n"
            "Referer: https://webapp.simplisafe.com/\r\n"
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36\r\n"
        )

        # Corrected, intentionally minimal audio-repair filter.
        #
        # async=1:
        #   Enable timestamp-driven audio compensation.
        #
        # min_hard_comp=0.100:
        #   Timestamp discrepancies >= 100 ms may use hard compensation
        #   (silence insertion / sample trimming) rather than only gradual
        #   resampling.
        #
        # first_pts=0:
        #   Start the repaired audio timeline at zero for this output session.
        audio_filter = "aresample=async=1:min_hard_comp=0.100:first_pts=0"

        cmd = [
            binary,
            "-hide_banner",
            "-loglevel",
            "warning",

            # Browser-like request headers for SimpliSafe's media endpoint.
            "-headers",
            headers,

            # SimpliSafe live FLV source.
            "-i",
            flv_url,

            # Explicitly select first video and optional first audio stream.
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",

            # Preserve video exactly; no video decode/re-encode.
            "-c:v",
            "copy",

            # Repair only the audio timeline.
            "-af",
            audio_filter,

            # Re-encode repaired audio to AAC for FLV compatibility.
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "64k",

            # Stay in FLV; do not return to MPEG-TS for this test.
            "-f",
            "flv",
            "pipe:1",
        ]

        _LOGGER.warning(
            "Starting SimpliRTC corrected audio-repair FLV proxy for %s: "
            "video=copy audio_filter=%s source=%s",
            entity_id,
            audio_filter,
            flv_url,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as err:
            _LOGGER.exception(
                "Failed to start FFmpeg corrected audio-repair proxy for %s",
                entity_id,
            )
            return web.Response(
                status=500,
                text=f"Failed to start FFmpeg: {err}",
            )

        _LOGGER.warning(
            "SimpliRTC corrected audio-repair FFmpeg started for %s, pid=%s",
            entity_id,
            proc.pid,
        )

        async def _log_stderr() -> None:
            assert proc.stderr is not None

            async for line in proc.stderr:
                text = line.decode(errors="replace").rstrip()
                if text:
                    _LOGGER.warning(
                        "SimpliRTC corrected audio-repair FFmpeg[%s]: %s",
                        entity_id,
                        text,
                    )

        stderr_task = self.hass.async_create_background_task(
            _log_stderr(),
            f"simplirtc-corrected-audio-repair-stderr-{entity_id}",
        )

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "video/x-flv",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

        try:
            await response.prepare(request)
        except Exception:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

            stderr_task.cancel()
            raise

        bytes_relayed = 0

        try:
            assert proc.stdout is not None

            while True:
                chunk = await proc.stdout.read(65536)

                if not chunk:
                    break

                bytes_relayed += len(chunk)
                await response.write(chunk)

        except (
            ConnectionResetError,
            BrokenPipeError,
            asyncio.CancelledError,
        ):
            _LOGGER.debug(
                "SimpliRTC corrected audio-repair FLV client "
                "disconnected for %s",
                entity_id,
            )

        except Exception:
            _LOGGER.exception(
                "Error relaying SimpliRTC corrected audio-repair FLV for %s",
                entity_id,
            )

        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

                try:
                    await proc.wait()
                except Exception:
                    pass

            stderr_task.cancel()

            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass

            _LOGGER.warning(
                "SimpliRTC corrected audio-repair FLV proxy ended for %s "
                "after %d bytes (returncode=%s)",
                entity_id,
                bytes_relayed,
                proc.returncode,
            )

        return response
