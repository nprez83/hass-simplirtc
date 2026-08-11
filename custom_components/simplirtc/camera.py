"""Component providing support for the SimpliSafe camera."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import logging
import secrets
import time
from typing import Any, TypeVar, override

import aiohttp
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass
from simplipy.device.camera import Camera, CameraTypes
from simplipy.system.v3 import SystemV3
from simplipy.websocket import EVENT_CAMERA_MOTION_DETECTED
import jwt

from homeassistant.config_entries import ConfigEntry
from homeassistant.components.camera import (
    Camera as CameraEntity,
    CameraEntityFeature,
    CameraEntityDescription,
    WebRTCAnswer,
    WebRTCCandidate,
    WebRTCClientConfiguration,
    WebRTCSendMessage,
)
from homeassistant.components.simplisafe import SimpliSafe
from homeassistant.components.simplisafe.entity import SimpliSafeEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from webrtc_models import RTCIceCandidateInit, RTCIceServer

from .kinesis import KinesisSession
from .livekit import LiveKitSession, fetch_ice_servers
from .protobufs.livekit_rtc_pb2 import (
    ICEServer,
    SessionDescription,
    SignalTarget,
    TrickleRequest,
)

_LOGGER = logging.getLogger(__name__)

WEBRTC_URL_BASE = "https://app-hub.prd.aser.simplisafe.com/v2"

# Preserve the original PR's startup debounce/settle behavior without
# calling the unsupported SimpliSafe camera-wakeup endpoint.
WAKE_DEBOUNCE_SECONDS = 10.0
WAKE_SETTLE_SECONDS = 5.0

GO2RTC_RTSP_PORT = 18554

_StreamResponseT = TypeVar("_StreamResponseT")


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry[SimpliSafe],
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up SimpliSafe cameras."""
    simplisafe = entry.runtime_data

    cameras: list[SimpliSafeCamera] = []

    for system in simplisafe.systems.values():
        if not isinstance(system, SystemV3):
            _LOGGER.warning(
                "Skipping camera setup for V%d system: %s",
                system.version,
                system.system_id,
            )
            continue

        for camera in system.cameras.values():
            if not isinstance(
                settings := camera.camera_settings.get("admin"),
                Mapping,
            ):
                _LOGGER.warning(
                    "Skipping camera '%s'. Unexpected settings schema.",
                    camera.name,
                )
                continue

            cls: type[SimpliSafeCamera]

            match settings.get("webRTCProvider"):
                case "mist":
                    cls = SimpliSafeLiveKitCamera

                case "kvs":
                    cls = SimpliSafeKenisisCamera

                case _ as provider:
                    # Cameras without a supported WebRTC backend, such as
                    # the video doorbell, use the legacy FLV endpoint.
                    if camera.camera_type in (
                        CameraTypes.CAMERA,
                        CameraTypes.DOORBELL,
                    ):
                        cls = SimpliSafeGo2rtcCamera
                    else:
                        _LOGGER.warning(
                            "Camera '%s' has unsupported backend "
                            "(provider=%s, type=%s)",
                            camera.name,
                            provider,
                            camera.camera_type,
                        )
                        continue

            cameras.append(cls(simplisafe, system, camera))

    async_add_entities(cameras)


@dataclass(kw_only=True, slots=True)
class KenisisResponse:
    """Kinesis live-view response."""

    signedChannelEndpoint: str
    clientId: str
    iceServers: list[Any]


@dataclass(kw_only=True, slots=True)
class LiveKitResponse:
    """LiveKit live-view response."""

    liveKitDetails: LiveKitDetails


@dataclass(kw_only=True, slots=True)
class LiveKitDetails:
    """LiveKit connection details."""

    liveKitURL: str
    userToken: str


class SimpliSafeCamera(SimpliSafeEntity, CameraEntity):
    """An implementation of a SimpliSafe camera."""

    def __init__(
        self,
        simplisafe: SimpliSafe,
        system: SystemV3,
        device: Camera,
    ) -> None:
        """Initialize the SimpliSafe camera."""
        super().__init__(
            simplisafe,
            system,
            device=device,
            additional_websocket_events=(EVENT_CAMERA_MOTION_DETECTED,),
        )

        self.entity_description = CameraEntityDescription(
            key="live_view",
        )

        CameraEntity.__init__(self)

        self._attr_unique_id = f"{super().unique_id}-camera"

        self._attr_supported_features |= CameraEntityFeature.STREAM

        self._device: Camera

        # Track the last stream-start settle cycle. The original PR called a
        # camera-wakeup endpoint here, but that endpoint returns 404 for this
        # system. We retain only the useful debounce/settle timing.
        self._last_wake_monotonic: float = 0.0

    async def _async_wake_cameras(self) -> bool:
        """Preserve the PR's startup debounce without calling wakeup API.

        Returns True when this is the first stream-start attempt outside the
        debounce window. The caller then waits WAKE_SETTLE_SECONDS before
        connecting go2rtc. No SimpliSafe HTTP request is made.
        """
        now = time.monotonic()

        if now - self._last_wake_monotonic < WAKE_DEBOUNCE_SECONDS:
            return False

        self._last_wake_monotonic = now
        return True

    async def _create_stream(
        self,
        response_type: type[_StreamResponseT],
    ) -> _StreamResponseT:
        """Request live-view connection information."""
        path = (
            f"cameras/{self._device.serial}/"
            f"{self._system.system_id}/live-view"
        )

        return TypeAdapter(response_type).validate_python(
            await self._simplisafe._api.async_request(
                "get",
                path,
                url_base=WEBRTC_URL_BASE,
            )
        )

    @override
    async def async_camera_image(
        self,
        width: int | None = None,
        height: int | None = None,
    ) -> bytes | None:
        """Return a camera image."""
        _ = width, height
        return None


class SimpliSafeGo2rtcCamera(SimpliSafeCamera):
    """SimpliSafe camera using the legacy FLV endpoint through go2rtc."""

    def __init__(
        self,
        simplisafe: SimpliSafe,
        system: SystemV3,
        device: Camera,
    ) -> None:
        """Initialize the go2rtc-backed SimpliSafe camera."""
        super().__init__(simplisafe, system, device)

        # Secret used to prevent arbitrary callers from using the local
        # authenticated FLV proxy.
        self._proxy_token = secrets.token_urlsafe(24)

    @property
    def proxy_token(self) -> str:
        """Return the secret guarding this camera's FLV proxy URL."""
        return self._proxy_token

    @property
    def access_token(self) -> str | None:
        """Return the current SimpliSafe bearer token."""
        return self._simplisafe._api.access_token

    def flv_url(self, width: int = 1280) -> str:
        """Return the SimpliSafe FLV media URL."""
        return self._device.video_url(width=width)

    @override
    async def stream_source(self) -> str | None:
        """Return the go2rtc RTSP URL for this camera."""

        if not self.access_token:
            _LOGGER.warning(
                "No SimpliSafe access token available for %s",
                self.entity_id,
            )
            return None

        # Preserve the original PR's 5-second pre-stream settle period, but
        # do not call the unsupported camera-wakeup endpoint. Rapid repeated
        # requests inside the debounce window skip the additional delay.
        if await self._async_wake_cameras():
            await asyncio.sleep(WAKE_SETTLE_SECONDS)

        port = self.hass.http.server_port

        proxy_url = (
            f"http://127.0.0.1:{port}"
            f"/api/simplirtc_flv/{self.entity_id}"
            f"?sig={self._proxy_token}"
        )

        # go2rtc's managed RTSP server receives a cleaned MPEG-TS stream
        # from the HA proxy. FFmpeg copies the H264 video and go2rtc handles
        # the final RTSP packaging.
        #go2rtc_source = (
        #    f"ffmpeg:{proxy_url}"
        #    "#video=copy"
        #    "#audio=opus"
        #)
        go2rtc_source = proxy_url

        _LOGGER.debug(
            "SimpliRTC go2rtc source for %s: %s",
            self.entity_id,
            go2rtc_source,
        )

        return await self._async_ensure_go2rtc_rtsp(go2rtc_source)

    async def _async_ensure_go2rtc_rtsp(
        self,
        source: str,
    ) -> str | None:
        """Publish the FLV source through go2rtc and return its RTSP URL."""
        try:
            from homeassistant.components.go2rtc.const import (
                DOMAIN as GO2RTC_DOMAIN,
            )

            entries = self.hass.config_entries.async_entries(
                GO2RTC_DOMAIN
            )

            if not entries:
                _LOGGER.error(
                    "SimpliRTC go2rtc RTSP: "
                    "no go2rtc config entry found"
                )
                return None

            client = entries[0].runtime_data._rest_client

            name = f"simplirtc_{self._device.serial}"

            _LOGGER.debug(
                "Registering go2rtc stream %s for %s",
                name,
                self.entity_id,
            )

            await client.streams.add(name, [source])

            rtsp_url = (
                f"rtsp://127.0.0.1:{GO2RTC_RTSP_PORT}/{name}"
            )

            _LOGGER.debug(
                "SimpliRTC go2rtc RTSP URL for %s: %s",
                self.entity_id,
                rtsp_url,
            )

            return rtsp_url

        except Exception as err:
            _LOGGER.error(
                "SimpliRTC go2rtc RTSP publish failed for %s: %r "
                "(source=%s)",
                self.entity_id,
                err,
                source,
            )
            return None

    @override
    async def async_camera_image(
        self,
        width: int | None = None,
        height: int | None = None,
    ) -> bytes | None:
        """Return a still image from the SimpliSafe MJPEG endpoint."""
        _ = height

        if not (token := self.access_token):
            return None

        snapshot_width = width or 1280

        media_base = self.flv_url(
            width=snapshot_width
        ).split("/flv?", 1)[0]

        url = (
            f"{media_base}/mjpg"
            f"?x={snapshot_width}&fr=1"
        )

        session = async_get_clientsession(self.hass)

        try:
            async with session.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                },
            ) as response:
                if response.status != 200:
                    _LOGGER.debug(
                        "Snapshot request for %s returned HTTP %s",
                        self.entity_id,
                        response.status,
                    )
                    return None

                return await response.read()

        except (
            aiohttp.ClientError,
            TimeoutError,
        ) as err:
            _LOGGER.debug(
                "Snapshot request for %s failed: %s",
                self.entity_id,
                err,
            )
            return None


class SimpliSafeLiveKitCamera(SimpliSafeCamera):
    """SimpliSafe camera using LiveKit WebRTC."""

    def __init__(
        self,
        simplisafe: SimpliSafe,
        system: SystemV3,
        device: Camera,
    ) -> None:
        super().__init__(simplisafe, system, device)

        self._livekit_url: str = ""
        self._livekit_token: str = ""
        self._livekit_ice_servers: list[ICEServer] = []

        self._livekit_client_configuration = (
            WebRTCClientConfiguration()
        )

        self._cache_expiration: float = 0

        self._lock = asyncio.Lock()

        self._ice_server_task: asyncio.Task[None] | None = None

        self._sessions: dict[str, LiveKitSession] = {}

    @override
    async def async_internal_added_to_hass(self) -> None:
        """Run when entity is added to Home Assistant."""
        await super().async_internal_added_to_hass()

        self._async_fetch_initial_livekit_ice_servers()

    @override
    async def async_will_remove_from_hass(self) -> None:
        """Run when entity is removed from Home Assistant."""
        if task := self._ice_server_task:
            self._ice_server_task = None
            task.cancel()

        await super().async_will_remove_from_hass()

    @override
    @callback
    def _async_get_webrtc_client_configuration(
        self,
    ) -> WebRTCClientConfiguration:
        """Return cached LiveKit ICE servers."""
        return self._livekit_client_configuration

    @callback
    def _async_fetch_initial_livekit_ice_servers(self) -> None:
        """Start the initial LiveKit ICE server fetch."""
        if (
            self._ice_server_task
            and not self._ice_server_task.done()
        ):
            return

        self._ice_server_task = self.hass.async_create_task(
            self._fetch_initial_livekit_ice_servers(),
            f"simplirtc-fetch-livekit-ice-{self.entity_id}",
        )

    async def _fetch_initial_livekit_ice_servers(self) -> None:
        """Fetch initial LiveKit ICE servers."""
        try:
            livekit_url, user_token = await self._live_view()

            self._async_update_livekit_ice_servers(
                await fetch_ice_servers(
                    livekit_url,
                    user_token,
                )
            )

        except Exception as err:
            _LOGGER.debug(
                "Failed to refresh LiveKit ICE servers for %s: %s",
                self.entity_id,
                err,
            )

        finally:
            if self._ice_server_task is asyncio.current_task():
                self._ice_server_task = None

    @callback
    def _async_update_livekit_ice_servers(
        self,
        ice_servers: list[ICEServer],
    ) -> None:
        """Store LiveKit ICE servers."""
        if self._livekit_ice_servers == ice_servers:
            return

        _LOGGER.debug(
            "Updating LiveKit ICE servers for %s: count=%s",
            self.entity_id,
            len(ice_servers),
        )

        self._livekit_ice_servers = ice_servers

        config = WebRTCClientConfiguration()

        for ice_server in ice_servers:
            config.configuration.ice_servers.append(
                RTCIceServer(
                    urls=list(ice_server.urls),
                    username=ice_server.username or None,
                    credential=ice_server.credential or None,
                )
            )

        self._livekit_client_configuration = config

    @override
    async def async_handle_async_webrtc_offer(
        self,
        offer_sdp: str,
        session_id: str,
        send_message: WebRTCSendMessage,
    ) -> None:
        """Handle a browser WebRTC offer through LiveKit."""
        livekit_url, user_token = await self._live_view()

        def send_answer(answer: SessionDescription) -> None:
            send_message(
                WebRTCAnswer(answer=answer.sdp)
            )

        def send_candidate(trickle: TrickleRequest) -> None:
            if trickle.final and not trickle.candidateInit:
                send_message(
                    WebRTCCandidate(
                        candidate=RTCIceCandidateInit(
                            candidate="",
                            sdp_mid=None,
                            sdp_m_line_index=None,
                        )
                    )
                )
                return

            if not trickle.candidateInit:
                return

            try:
                candidate_init = json.loads(
                    trickle.candidateInit
                )
            except ValueError as err:
                _LOGGER.warning(
                    "Dropping invalid LiveKit ICE candidate JSON: %s",
                    err,
                )
                return

            candidate = candidate_init.get("candidate")

            if not isinstance(candidate, str):
                _LOGGER.warning(
                    "Dropping LiveKit ICE candidate "
                    "without candidate field"
                )
                return

            sdp_mid = candidate_init.get("sdpMid")
            sdp_m_line_index = candidate_init.get(
                "sdpMLineIndex"
            )

            send_message(
                WebRTCCandidate(
                    candidate=RTCIceCandidateInit(
                        candidate=candidate,
                        sdp_mid=(
                            sdp_mid
                            if isinstance(sdp_mid, str)
                            else None
                        ),
                        sdp_m_line_index=(
                            sdp_m_line_index
                            if isinstance(
                                sdp_m_line_index,
                                int,
                            )
                            else None
                        ),
                    )
                )
            )

        def on_close() -> None:
            if self._sessions.get(session_id) is session:
                self._sessions.pop(session_id, None)

        session = LiveKitSession(
            session_id=session_id,
            livekit_url=livekit_url,
            user_token=user_token,
            offer_sdp=offer_sdp,
            send_answer=send_answer,
            send_candidate=send_candidate,
            on_close=on_close,
            on_ice_servers=self._async_update_livekit_ice_servers,
        )

        self._sessions[session_id] = session

        try:
            await session.start()
        except BaseException:
            on_close()
            raise

    @override
    async def async_on_webrtc_candidate(
        self,
        session_id: str,
        candidate: RTCIceCandidateInit,
    ) -> None:
        """Handle a WebRTC candidate for LiveKit."""
        if not (
            session := self._sessions.get(session_id)
        ):
            _LOGGER.debug(
                "Ignoring WebRTC candidate for "
                "closed session %s",
                session_id,
            )
            return

        if not candidate.candidate:
            return

        candidate_init: dict[str, str | int] = {
            "candidate": candidate.candidate
        }

        if candidate.sdp_mid is not None:
            candidate_init["sdpMid"] = candidate.sdp_mid

        if candidate.sdp_m_line_index is not None:
            candidate_init["sdpMLineIndex"] = (
                candidate.sdp_m_line_index
            )

        await session.send_candidate(
            TrickleRequest(
                candidateInit=json.dumps(
                    candidate_init,
                    separators=(",", ":"),
                ),
                target=SignalTarget.PUBLISHER,
            )
        )

    @override
    @callback
    def close_webrtc_session(
        self,
        session_id: str,
    ) -> None:
        """Close a LiveKit signaling session."""
        if session := self._sessions.pop(
            session_id,
            None,
        ):
            session.close()

    async def _live_view(self) -> tuple[str, str]:
        """Return cached LiveKit connection information."""
        if time.time() < self._cache_expiration:
            return (
                self._livekit_url,
                self._livekit_token,
            )

        async with self._lock:
            if time.time() < self._cache_expiration:
                return (
                    self._livekit_url,
                    self._livekit_token,
                )

            live_view = await self._create_stream(
                LiveKitResponse
            )

            self._livekit_url = (
                live_view.liveKitDetails.liveKitURL
            )

            self._livekit_token = (
                live_view.liveKitDetails.userToken
            )

            try:
                decoded_token = jwt.decode(
                    self._livekit_token,
                    options={
                        "verify_signature": False
                    },
                )

                self._cache_expiration = decoded_token["exp"]

            except Exception as err:
                _LOGGER.warning(
                    "Failed to decode JWT token for caching: %s",
                    err,
                )
                self._cache_expiration = 0

        return (
            self._livekit_url,
            self._livekit_token,
        )


class SimpliSafeKenisisCamera(SimpliSafeCamera):
    """SimpliSafe camera using Kinesis WebRTC."""

    def __init__(
        self,
        simplisafe: SimpliSafe,
        system: SystemV3,
        device: Camera,
    ) -> None:
        super().__init__(
            simplisafe,
            system,
            device,
        )

        self._sessions: dict[
            str,
            asyncio.Task[KinesisSession],
        ] = {}

    @override
    async def async_handle_async_webrtc_offer(
        self,
        offer_sdp: str,
        session_id: str,
        send_message: WebRTCSendMessage,
    ) -> None:
        """Handle a Kinesis WebRTC offer."""
        self._sessions[session_id] = (
            session_future
        ) = self.hass.async_create_task(
            self._create_webrtc_session(
                session_id,
                send_message,
            )
        )

        try:
            session = await session_future

            if self._sessions.get(session_id) is session_future:
                session.start(offer_sdp)

        except Exception:
            self._sessions.pop(
                session_id,
                None,
            )
            raise

    @override
    async def async_on_webrtc_candidate(
        self,
        session_id: str,
        candidate: RTCIceCandidateInit,
    ) -> None:
        """Handle a Kinesis WebRTC candidate."""
        if not (
            session_future := self._sessions.get(
                session_id
            )
        ):
            _LOGGER.debug(
                "Ignoring WebRTC candidate for "
                "closed session %s",
                session_id,
            )
            return

        try:
            session = await session_future

        except Exception as err:
            _LOGGER.debug(
                "Ignoring WebRTC candidate for "
                "failed session %s: %s",
                session_id,
                err,
            )
            return

        await session.send_candidate(
            candidate.candidate,
            sdp_mid=candidate.sdp_mid,
            sdp_m_line_index=(
                candidate.sdp_m_line_index
            ),
        )

    @override
    @callback
    def close_webrtc_session(
        self,
        session_id: str,
    ) -> None:
        """Close a Kinesis WebRTC session."""
        if not (
            session_future := self._sessions.pop(
                session_id,
                None,
            )
        ):
            return

        async def close_session() -> None:
            if not session_future.done():
                session_future.cancel()

            try:
                session = await session_future

            except asyncio.CancelledError:
                return

            except Exception as err:
                _LOGGER.debug(
                    "WebRTC session %s ended before "
                    "startup completed: %s",
                    session_id,
                    err,
                )
                return

            session.close()

        self.hass.async_create_task(
            close_session()
        )

    async def _create_webrtc_session(
        self,
        session_id: str,
        send_message: WebRTCSendMessage,
    ) -> KinesisSession:
        """Create a Kinesis WebRTC session."""
        live_view = await self._create_stream(
            KenisisResponse
        )

        def send_candidate(
            candidate: str,
            sdp_mid: str | None,
            sdp_m_line_index: int | None,
        ) -> None:
            send_message(
                WebRTCCandidate(
                    candidate=RTCIceCandidateInit(
                        candidate=candidate,
                        sdp_mid=sdp_mid,
                        sdp_m_line_index=sdp_m_line_index,
                    )
                )
            )

        return KinesisSession(
            session_id=session_id,
            channel_endpoint=(
                live_view.signedChannelEndpoint
            ),
            client_id=live_view.clientId,
            send_answer=lambda answer_sdp: send_message(
                WebRTCAnswer(answer=answer_sdp)
            ),
            send_candidate=send_candidate,
        )
