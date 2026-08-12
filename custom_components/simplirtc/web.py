"""Web views for the SimpliRTC integration."""

from __future__ import annotations

import asyncio
import logging
import time

from aiohttp import web

from homeassistant.components.camera import DATA_COMPONENT
from homeassistant.components.ffmpeg import get_ffmpeg_manager
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

from .camera import SimpliSafeGo2rtcCamera, SimpliSafeLiveKitCamera
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

STREAM_WIDTH = 640

# Protective circuit-breaker settings.
FAILURE_COOLDOWN_SECONDS = 30.0
SESSION_COOLDOWN_SECONDS = 15.0

# Bounded automatic recovery after a clean established upstream EOF.
AUTO_RECOVERY_DELAY_SECONDS = 15.0
AUTO_RECOVERY_WINDOW_SECONDS = 20.0

# A recovered session must prove healthy before earning another recovery.
HEALTHY_SESSION_MIN_SECONDS = 30.0
HEALTHY_SESSION_MIN_BYTES = 1_000_000

# Runtime-only state; cleared on Home Assistant restart.
_CAMERA_STATE: dict[str, dict[str, object]] = {}


def _state_for(entity_id: str) -> dict[str, object]:
    state = _CAMERA_STATE.get(entity_id)
    if state is None:
        state = {
            "active": False,
            "cooldown_until": 0.0,
            "cooldown_kind": None,
            "recovery_available": False,
            "recovery_not_before": 0.0,
            "recovery_until": 0.0,
            "recovery_used": False,
            "session_started_at": 0.0,
            "session_was_auto_recovery": False,
        }
        _CAMERA_STATE[entity_id] = state
    return state


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
    """Proxy SimpliSafe FLV with corrected AAC audio and camera protection.

    Media path is unchanged from the corrected-AAC baseline:
      * H.264 video is stream-copied unchanged.
      * AAC audio is decoded, timestamp-repaired with aresample, then
        re-encoded to AAC.
      * Output remains FLV for go2rtc native HTTP/FLV ingest.
      * No automatic reconnect loop is added here.

    Protection added by this test:
      * Only one upstream FFmpeg session may run per camera entity.
      * Duplicate local requests are rejected without contacting SimpliSafe.
      * Every real upstream start requires a one-shot lease granted by
        camera.stream_source().
      * go2rtc reconnects after EOF are rejected locally once that lease has
        been consumed.
      * Zero-byte / failed sessions trigger a 30-second local cooldown.
      * Completed media sessions trigger a 15-second local cooldown.
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

        state = _state_for(entity_id)
        now = time.monotonic()

        # Reject near-simultaneous local requests before they can launch
        # additional upstream FFmpeg/SimpliSafe sessions.
        if bool(state["active"]):
            _LOGGER.warning(
                "Protected SimpliRTC: rejecting duplicate request for %s; "
                "upstream session already active; no SimpliSafe request made",
                entity_id,
            )
            return web.Response(
                status=503,
                headers={"Retry-After": "5"},
                text="SimpliRTC upstream session already active",
            )

        cooldown_until = float(state["cooldown_until"])
        cooldown_kind = state.get("cooldown_kind")
        lease_consumed = False
        auto_recovery = False

        recovery_available = bool(state.get("recovery_available", False))
        recovery_not_before = float(state.get("recovery_not_before", 0.0))
        recovery_until = float(state.get("recovery_until", 0.0))

        if recovery_available and now > recovery_until:
            state["recovery_available"] = False
            recovery_available = False
            _LOGGER.warning(
                "SIMPLIRTC Protected: bounded auto-recovery window expired "
                "for %s; fresh HA lease now required",
                entity_id,
            )

        if now < cooldown_until:
            remaining = max(1, int(cooldown_until - now + 0.999))

            if cooldown_kind == "completed-session":
                if camera.consume_stream_start_lease():
                    lease_consumed = True
                    state["cooldown_until"] = 0.0
                    state["cooldown_kind"] = None
                    state["recovery_available"] = False
                    state["recovery_used"] = False
                    _LOGGER.warning(
                        "SIMPLIRTC Protected: fresh lease bypassing "
                        "completed-session cooldown for %s (%ds remained)",
                        entity_id,
                        remaining,
                    )
                elif (
                    recovery_available
                    and recovery_not_before <= now <= recovery_until
                ):
                    auto_recovery = True
                    state["recovery_available"] = False
                    state["recovery_used"] = True
                    state["cooldown_until"] = 0.0
                    state["cooldown_kind"] = None
                    _LOGGER.warning(
                        "SIMPLIRTC Protected: allowing ONE bounded "
                        "automatic recovery for %s after clean upstream EOF",
                        entity_id,
                    )
                else:
                    if recovery_available and now < recovery_not_before:
                        wait = max(
                            1,
                            int(recovery_not_before - now + 0.999),
                        )
                        retry_after = min(wait, remaining)
                        _LOGGER.warning(
                            "SIMPLIRTC Protected: delaying bounded automatic "
                            "recovery for %s for %ds after clean EOF; "
                            "no SimpliSafe request made yet",
                            entity_id,
                            wait,
                        )
                    else:
                        retry_after = remaining
                        _LOGGER.warning(
                            "Protected SimpliRTC: completed-session cooldown "
                            "blocking unleased request for %s for %ds; "
                            "no SimpliSafe request made",
                            entity_id,
                            remaining,
                        )

                    return web.Response(
                        status=503,
                        headers={"Retry-After": str(retry_after)},
                        text=(
                            "SimpliRTC completed-session cooldown active "
                            f"({remaining}s remaining)"
                        ),
                    )
            else:
                state["recovery_available"] = False
                discarded_lease = camera.consume_stream_start_lease()

                if discarded_lease:
                    _LOGGER.warning(
                        "SIMPLIRTC Protected: failure cooldown rejected and "
                        "discarded fresh lease for %s; %ds remain; "
                        "no SimpliSafe request made",
                        entity_id,
                        remaining,
                    )
                else:
                    _LOGGER.warning(
                        "Protected SimpliRTC: failure cooldown blocking %s "
                        "for %ds; no SimpliSafe request made",
                        entity_id,
                        remaining,
                    )

                return web.Response(
                    status=503,
                    headers={"Retry-After": str(remaining)},
                    text=(
                        "SimpliRTC failure cooldown active "
                        f"({remaining}s remaining)"
                    ),
                )

        if not lease_consumed and not auto_recovery:
            if camera.consume_stream_start_lease():
                lease_consumed = True
                state["recovery_available"] = False
                state["recovery_used"] = False
            else:
                recovery_available = bool(
                    state.get("recovery_available", False)
                )
                recovery_not_before = float(
                    state.get("recovery_not_before", 0.0)
                )
                recovery_until = float(state.get("recovery_until", 0.0))

                if (
                    recovery_available
                    and recovery_not_before <= now <= recovery_until
                ):
                    auto_recovery = True
                    state["recovery_available"] = False
                    state["recovery_used"] = True
                    _LOGGER.warning(
                        "SIMPLIRTC Protected: allowing ONE bounded "
                        "automatic recovery for %s after clean upstream EOF",
                        entity_id,
                    )
                else:
                    _LOGGER.warning(
                        "Protected SimpliRTC: rejecting unleased upstream "
                        "request for %s; likely go2rtc reconnect/probe; "
                        "no SimpliSafe request made",
                        entity_id,
                    )
                    return web.Response(
                        status=503,
                        headers={"Retry-After": "5"},
                        text="SimpliRTC upstream start not authorized",
                    )

        if auto_recovery:
            _LOGGER.warning(
                "SIMPLIRTC Protected: bounded automatic recovery starting "
                "for %s",
                entity_id,
            )

        # Claim the upstream slot before starting FFmpeg.
        state["active"] = True
        state["session_started_at"] = time.monotonic()
        state["session_was_auto_recovery"] = auto_recovery

        proc = None
        stderr_task = None
        bytes_relayed = 0

        try:
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

            audio_filter = "aresample=async=1:min_hard_comp=0.100:first_pts=0"

            cmd = [
                binary,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-headers",
                headers,
                "-i",
                flv_url,
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "copy",
                "-af",
                audio_filter,
                "-c:a",
                "aac",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-b:a",
                "64k",
                "-f",
                "flv",
                "pipe:1",
            ]

            _LOGGER.warning(
                "Starting LEASE-PROTECTED SimpliRTC corrected audio-repair FLV proxy "
                "for %s: video=copy audio_filter=%s source=%s",
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
                state["cooldown_until"] = (
                    time.monotonic() + FAILURE_COOLDOWN_SECONDS
                )
                state["cooldown_kind"] = "failure"
                state["recovery_available"] = False
                _LOGGER.exception(
                    "Failed to start protected FFmpeg for %s; "
                    "%.0fs failure cooldown armed",
                    entity_id,
                    FAILURE_COOLDOWN_SECONDS,
                )
                return web.Response(
                    status=500,
                    text=f"Failed to start FFmpeg: {err}",
                )

            _LOGGER.warning(
                "Lease-protected SimpliRTC FFmpeg started for %s, pid=%s",
                entity_id,
                proc.pid,
            )

            async def _log_stderr() -> None:
                assert proc is not None
                assert proc.stderr is not None

                async for line in proc.stderr:
                    line_text = line.decode(errors="replace").rstrip()
                    if line_text:
                        _LOGGER.warning(
                            "Protected SimpliRTC FFmpeg[%s]: %s",
                            entity_id,
                            line_text,
                        )

            stderr_task = self.hass.async_create_background_task(
                _log_stderr(),
                f"simplirtc-protected-audio-repair-stderr-{entity_id}",
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

                if stderr_task is not None:
                    stderr_task.cancel()

                state["cooldown_until"] = (
                    time.monotonic() + FAILURE_COOLDOWN_SECONDS
                )
                state["cooldown_kind"] = "failure"
                state["recovery_available"] = False
                raise

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
                    "Protected SimpliRTC FLV client disconnected for %s",
                    entity_id,
                )

            except Exception:
                _LOGGER.exception(
                    "Error relaying protected SimpliRTC FLV for %s",
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

                if stderr_task is not None:
                    stderr_task.cancel()
                    try:
                        await stderr_task
                    except (asyncio.CancelledError, Exception):
                        pass

            # Arm cooldown AFTER the upstream process is fully gone, so no
            # local request can slip through between process exit and state
            # update.
            ended_at = time.monotonic()
            session_started_at = float(state.get("session_started_at", 0.0))
            session_runtime = (
                max(0.0, ended_at - session_started_at)
                if session_started_at > 0.0
                else 0.0
            )
            session_was_auto_recovery = bool(
                state.get("session_was_auto_recovery", False)
            )

            if bytes_relayed == 0 or proc.returncode not in (0, -9):
                cooldown = FAILURE_COOLDOWN_SECONDS
                reason = "failure/zero-byte"
                state["recovery_available"] = False

            elif proc.returncode == -9:
                # Normal local cleanup because downstream viewer disappeared.
                # Never auto-recover a user/navigation teardown.
                cooldown = SESSION_COOLDOWN_SECONDS
                reason = "completed-session"
                state["recovery_available"] = False
                state["recovery_used"] = False

            else:
                # Clean established upstream EOF.
                cooldown = SESSION_COOLDOWN_SECONDS
                reason = "completed-session"

                healthy_session = (
                    session_runtime >= HEALTHY_SESSION_MIN_SECONDS
                    and bytes_relayed >= HEALTHY_SESSION_MIN_BYTES
                )

                # A recovered session that proves healthy earns a fresh
                # recovery allowance. This permits long continuous viewing
                # across repeated clean upstream EOFs without allowing rapid
                # restart loops.
                if session_was_auto_recovery and healthy_session:
                    state["recovery_used"] = False
                    _LOGGER.warning(
                        "SIMPLIRTC Protected: recovered session for %s "
                        "proved healthy (%.1fs, %d bytes); "
                        "automatic recovery allowance renewed",
                        entity_id,
                        session_runtime,
                        bytes_relayed,
                    )

                if not bool(state.get("recovery_used", False)):
                    state["recovery_available"] = True
                    state["recovery_not_before"] = (
                        ended_at + AUTO_RECOVERY_DELAY_SECONDS
                    )
                    state["recovery_until"] = (
                        ended_at
                        + AUTO_RECOVERY_DELAY_SECONDS
                        + AUTO_RECOVERY_WINDOW_SECONDS
                    )
                    _LOGGER.warning(
                        "SIMPLIRTC Protected: clean upstream EOF for %s; "
                        "bounded auto-recovery armed in %.0fs "
                        "for a %.0fs window",
                        entity_id,
                        AUTO_RECOVERY_DELAY_SECONDS,
                        AUTO_RECOVERY_WINDOW_SECONDS,
                    )
                else:
                    state["recovery_available"] = False
                    _LOGGER.warning(
                        "SIMPLIRTC Protected: clean upstream EOF for %s "
                        "before recovered session proved healthy; "
                        "fresh HA lease required",
                        entity_id,
                    )

            state["cooldown_until"] = ended_at + cooldown
            state["cooldown_kind"] = reason

            _LOGGER.warning(
                "Protected SimpliRTC FLV proxy ended for %s after %d bytes "
                "(returncode=%s); %.0fs %s cooldown armed",
                entity_id,
                bytes_relayed,
                proc.returncode,
                cooldown,
                reason,
            )

            return response

        finally:
            # Always release the active slot, even on an unexpected exception.
            state["active"] = False
