from pydantic import BaseModel


class CameraMetadata(BaseModel):
    lat: float
    lon: float
    bearing_deg: float   # compass bearing camera points (0=North, 90=East)
    fov_deg: float       # horizontal field of view in degrees


class DevicePayload(BaseModel):
    device_id: str
    timestamp: float
    rtsp_url: str        # e.g. rtsp://192.168.1.x:8554/cam_01
    camera: CameraMetadata
