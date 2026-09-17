from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("FOOD_MODEL_PATH", str(BASE_DIR / "모델" / "food_best.pt")))
STOOL_MODEL_PATH = Path(os.getenv("STOOL_MODEL_PATH", str(BASE_DIR / "모델" / "stool_best.pt")))
SKIN_MODEL_PATH = Path(os.getenv("SKIN_MODEL_PATH", str(BASE_DIR / "모델" / "skin_best.pt")))
EYE_MODEL_PATH = Path(os.getenv("EYE_MODEL_PATH", str(BASE_DIR / "모델" / "eye_best.pt")))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
RULES_PATH = BASE_DIR / "음식_위험_규칙.json"
STOOL_RULES_PATH = BASE_DIR / "배변_판정_규칙.json"

app = FastAPI(title="꼬리신호 음식·배변 판별 API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def load_rules() -> dict[str, Any]:
    try:
        return json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"danger_classes": {}, "safe_classes": {}}


def load_stool_rules() -> dict[str, Any]:
    try:
        return json.loads(STOOL_RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"classes": {}}


RULES = load_rules()
STOOL_RULES = load_stool_rules()
MODEL = None
MODEL_ERROR = None
STOOL_MODEL = None
STOOL_MODEL_ERROR = None

if YOLO is not None and MODEL_PATH.exists():
    try:
        MODEL = YOLO(str(MODEL_PATH))
    except Exception as exc:  # 모델 파일이 잘못되어도 API는 상태를 반환하며 시작
        MODEL_ERROR = str(exc)
elif YOLO is None:
    MODEL_ERROR = "ultralytics 패키지가 설치되지 않았습니다."
else:
    MODEL_ERROR = f"모델 파일이 없습니다: {MODEL_PATH}"

if YOLO is not None and STOOL_MODEL_PATH.exists():
    try:
        STOOL_MODEL = YOLO(str(STOOL_MODEL_PATH))
    except Exception as exc:  # 모델 파일이 잘못되어도 API는 상태를 반환하며 시작
        STOOL_MODEL_ERROR = str(exc)
elif YOLO is None:
    STOOL_MODEL_ERROR = "ultralytics 패키지가 설치되지 않았습니다."
else:
    STOOL_MODEL_ERROR = f"배변 모델 파일이 없습니다: {STOOL_MODEL_PATH}"


def load_optional_yolo_model(path: Path, label: str):
    if YOLO is None:
        return None, "ultralytics 패키지가 설치되지 않았습니다."
    if not path.exists():
        return None, f"{label} 모델 파일이 없습니다: {path}"
    try:
        return YOLO(str(path)), None
    except Exception as exc:
        return None, str(exc)


SKIN_MODEL, SKIN_MODEL_ERROR = load_optional_yolo_model(SKIN_MODEL_PATH, "피부")
EYE_MODEL, EYE_MODEL_ERROR = load_optional_yolo_model(EYE_MODEL_PATH, "안구")


def normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("_", "-")


def lookup_risk(class_name: str) -> tuple[str, str]:
    key = normalize(class_name)
    for name, info in RULES.get("danger_classes", {}).items():
        aliases = [name, *info.get("aliases", [])]
        if any(normalize(alias) == key for alias in aliases):
            return "danger", info.get("message", "강아지에게 위험할 수 있는 음식입니다.")
    for name, info in RULES.get("safe_classes", {}).items():
        aliases = [name, *info.get("aliases", [])]
        if any(normalize(alias) == key for alias in aliases):
            return "check", info.get("message", "개별 성분과 섭취량을 확인해 주세요.")
    return "unknown", "음식 종류를 확인했지만 안전 여부를 확정하지 못했습니다. 성분표를 확인해 주세요."


def lookup_stool(class_name: str) -> tuple[str, str, str]:
    key = normalize(class_name)
    for name, info in STOOL_RULES.get("classes", {}).items():
        aliases = [name, *info.get("aliases", [])]
        if any(normalize(alias) == key for alias in aliases):
            return (
                info.get("level", "observe"),
                info.get("message", "배변 상태를 관찰해 주세요."),
                info.get("recommendation", "횟수와 색, 식사량·활력 변화를 함께 기록해 주세요."),
            )
    return (
        "unknown",
        "배변 상태를 확실히 분류하지 못했습니다.",
        "사진 결과만으로 판단하지 말고 색·횟수·혈액이나 점액 여부와 강아지의 활력을 함께 확인해 주세요.",
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "food_model": {
            "model_ready": MODEL is not None,
            "model_path": str(MODEL_PATH),
            "model_error": MODEL_ERROR,
        },
        "stool_model": {
            "model_ready": STOOL_MODEL is not None,
            "model_path": str(STOOL_MODEL_PATH),
            "model_error": STOOL_MODEL_ERROR,
        },
        "skin_model": {
            "model_ready": SKIN_MODEL is not None,
            "model_path": str(SKIN_MODEL_PATH),
            "model_error": SKIN_MODEL_ERROR,
        },
        "eye_model": {
            "model_ready": EYE_MODEL is not None,
            "model_path": str(EYE_MODEL_PATH),
            "model_error": EYE_MODEL_ERROR,
        },
        "maps_api": {"configured": bool(GOOGLE_MAPS_API_KEY)},
    }


@app.post("/api/food-check")
async def food_check(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"status": "invalid_file", "message": "이미지 파일만 업로드해 주세요."}, status_code=400)
    if MODEL is None:
        return JSONResponse(
            {
                "status": "model_missing",
                "model_ready": False,
                "message": "학습된 음식 판별 모델을 모델/food_best.pt에 넣어 주세요.",
                "detail": MODEL_ERROR,
            },
            status_code=503,
        )

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        result = MODEL.predict(source=image, imgsz=224, verbose=False)[0]
        if result.probs is None:
            return JSONResponse(
                {"status": "wrong_model_task", "message": "Classification 모델(best-cls.pt)을 사용해 주세요."},
                status_code=422,
            )
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = str(result.names[class_id])
        if confidence < 0.60:
            risk, message = "unknown", "음식을 확실히 판별하지 못했습니다. 음식명과 성분을 직접 확인해 주세요."
        else:
            risk, message = lookup_risk(class_name)
        return {
            "status": "ok",
            "model_ready": True,
            "food": class_name,
            "confidence": round(confidence, 3),
            "risk": risk,
            "message": message,
            "medical_notice": "사진 판별은 참고용이며 수의사의 진료나 중독 상담을 대신하지 않습니다.",
        }
    except Exception as exc:
        return JSONResponse({"status": "inference_error", "message": "사진을 판별하지 못했습니다.", "detail": str(exc)}, status_code=500)


@app.post("/api/stool-check")
async def stool_check(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"status": "invalid_file", "message": "이미지 파일만 업로드해 주세요."}, status_code=400)
    if STOOL_MODEL is None:
        return JSONResponse(
            {
                "status": "model_missing",
                "model_ready": False,
                "message": "학습된 배변 판별 모델을 모델/stool_best.pt에 넣어 주세요.",
                "detail": STOOL_MODEL_ERROR,
            },
            status_code=503,
        )

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        result = STOOL_MODEL.predict(source=image, imgsz=224, verbose=False)[0]
        if result.probs is None:
            return JSONResponse(
                {"status": "wrong_model_task", "message": "배변 Classification 모델(best-cls.pt)을 사용해 주세요."},
                status_code=422,
            )
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = str(result.names[class_id])
        if confidence < 0.60:
            level = "unknown"
            message = "배변 상태를 확실히 판별하지 못했습니다."
            recommendation = "사진만으로 판단하지 말고 배변 횟수·색·혈액 또는 점액 여부와 활력 변화를 확인해 주세요."
        else:
            level, message, recommendation = lookup_stool(class_name)
        return {
            "status": "ok",
            "model_ready": True,
            "stool": class_name,
            "confidence": round(confidence, 3),
            "level": level,
            "message": message,
            "recommendation": recommendation,
            "medical_notice": "사진 분류는 관찰용 참고이며 수의사의 진단이나 치료를 대신하지 않습니다.",
        }
    except Exception as exc:
        return JSONResponse({"status": "inference_error", "message": "배변 사진을 판별하지 못했습니다.", "detail": str(exc)}, status_code=500)


@app.post("/api/pet-health-check")
async def pet_health_check(type: str, file: UploadFile = File(...)) -> JSONResponse:
    if type not in {"skin", "eye"}:
        return JSONResponse({"status": "invalid_type", "message": "type은 skin 또는 eye여야 합니다."}, status_code=400)
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"status": "invalid_file", "message": "이미지 파일만 업로드해 주세요."}, status_code=400)

    model = SKIN_MODEL if type == "skin" else EYE_MODEL
    model_error = SKIN_MODEL_ERROR if type == "skin" else EYE_MODEL_ERROR
    label = "피부" if type == "skin" else "안구"
    if model is None:
        return JSONResponse(
            {
                "status": "model_missing",
                "model_ready": False,
                "type": type,
                "message": f"학습된 {label} 상태 모델을 연결해 주세요.",
                "detail": model_error,
            },
            status_code=503,
        )

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        result = model.predict(source=image, imgsz=224, verbose=False)[0]
        if result.probs is None:
            return JSONResponse(
                {"status": "wrong_model_task", "message": f"{label} 분류 모델(best-cls.pt)을 사용해 주세요."},
                status_code=422,
            )
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = str(result.names[class_id])
        message = (
            f"{label} 상태 후보를 확인했습니다."
            if confidence >= 0.60
            else f"{label} 상태를 확실히 분류하지 못했습니다."
        )
        return {
            "status": "ok",
            "model_ready": True,
            "type": type,
            "condition": class_name if confidence >= 0.60 else "판별 불가",
            "confidence": round(confidence, 3),
            "message": message,
            "recommendation": "사진 결과는 참고용이며 증상이 있으면 수의사 진료를 받아 주세요.",
            "medical_notice": "사진 분류는 진단이나 치료를 대신하지 않습니다.",
        }
    except Exception as exc:
        return JSONResponse({"status": "inference_error", "message": f"{label} 사진을 판별하지 못했습니다.", "detail": str(exc)}, status_code=500)


@app.get("/api/nearby-vets")
async def nearby_vets(lat: float, lng: float, radius: int = 5000) -> JSONResponse:
    radius = max(500, min(radius, 50000))
    maps_url = "https://www.google.com/maps/search/?api=1&query=" + f"%EB%8F%99%EB%AC%BC%EB%B3%91%EC%9B%90+{lat},{lng}"
    if not GOOGLE_MAPS_API_KEY:
        return JSONResponse(
            {
                "status": "maps_api_missing",
                "message": "Google Places API 키가 아직 설정되지 않았습니다.",
                "maps_url": maps_url,
                "places": [],
            },
            status_code=503,
        )

    payload = {
        "includedTypes": ["veterinary_care"],
        "maxResultCount": 10,
        "rankPreference": "DISTANCE",
        "languageCode": "ko",
        "regionCode": "KR",
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius),
            }
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.nationalPhoneNumber,places.googleMapsUri,places.currentOpeningHours,places.regularOpeningHours,places.rating,places.userRatingCount",
    }
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.post("https://places.googleapis.com/v1/places:searchNearby", json=payload, headers=headers)
        if response.status_code >= 400:
            return JSONResponse(
                {"status": "maps_api_error", "message": "주변 동물병원 검색 API가 응답하지 않았습니다.", "detail": response.text, "maps_url": maps_url, "places": []},
                status_code=502,
            )
        data = response.json()
        places = []
        for place in data.get("places", []):
            places.append(
                {
                    "id": place.get("id", ""),
                    "name": place.get("displayName", {}).get("text", "동물병원"),
                    "address": place.get("formattedAddress", "주소 정보 없음"),
                    "phone": place.get("nationalPhoneNumber", "전화번호 정보 없음"),
                    "maps_url": place.get("googleMapsUri", maps_url),
                    "location": place.get("location", {}),
                    "opening_hours": place.get("currentOpeningHours", {}).get("weekdayDescriptions", []),
                    "open_now": place.get("currentOpeningHours", {}).get("openNow"),
                    "regular_opening_hours": place.get("regularOpeningHours", {}).get("weekdayDescriptions", []),
                    "rating": place.get("rating"),
                    "user_rating_count": place.get("userRatingCount"),
                }
            )
        return {"status": "ok", "places": places, "maps_url": maps_url}
    except Exception as exc:
        return JSONResponse(
            {"status": "maps_api_error", "message": "주변 동물병원 검색 중 오류가 발생했습니다.", "detail": str(exc), "maps_url": maps_url, "places": []},
            status_code=502,
        )


@app.get("/api/search-vets")
async def search_vets(query: str) -> JSONResponse:
    query = query.strip()[:120]
    maps_url = "https://www.google.com/maps/search/?api=1&query=" + quote(f"동물병원 {query}")
    if not query:
        return JSONResponse({"status": "invalid_query", "message": "검색할 지역을 입력해 주세요.", "maps_url": maps_url, "places": []}, status_code=400)
    if not GOOGLE_MAPS_API_KEY:
        return JSONResponse(
            {
                "status": "maps_api_missing",
                "message": "Google Places API 키가 아직 설정되지 않았습니다.",
                "maps_url": maps_url,
                "places": [],
            },
            status_code=503,
        )

    payload = {
        "textQuery": f"동물병원 {query}",
        "includedType": "veterinary_care",
        "pageSize": 10,
        "languageCode": "ko",
        "regionCode": "KR",
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.nationalPhoneNumber,places.googleMapsUri,places.currentOpeningHours,places.regularOpeningHours,places.rating,places.userRatingCount",
    }
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.post("https://places.googleapis.com/v1/places:searchText", json=payload, headers=headers)
        if response.status_code >= 400:
            return JSONResponse(
                {"status": "maps_api_error", "message": "입력한 지역의 동물병원 검색 API가 응답하지 않았습니다.", "detail": response.text, "maps_url": maps_url, "places": []},
                status_code=502,
            )
        data = response.json()
        places = []
        for place in data.get("places", []):
            places.append(
                {
                    "id": place.get("id", ""),
                    "name": place.get("displayName", {}).get("text", "동물병원"),
                    "address": place.get("formattedAddress", "주소 정보 없음"),
                    "phone": place.get("nationalPhoneNumber", "전화번호 정보 없음"),
                    "maps_url": place.get("googleMapsUri", maps_url),
                    "location": place.get("location", {}),
                    "opening_hours": place.get("currentOpeningHours", {}).get("weekdayDescriptions", []),
                    "open_now": place.get("currentOpeningHours", {}).get("openNow"),
                    "regular_opening_hours": place.get("regularOpeningHours", {}).get("weekdayDescriptions", []),
                    "rating": place.get("rating"),
                    "user_rating_count": place.get("userRatingCount"),
                }
            )
        return {"status": "ok", "places": places, "maps_url": maps_url}
    except Exception as exc:
        return JSONResponse(
            {"status": "maps_api_error", "message": "입력한 지역의 동물병원 검색 중 오류가 발생했습니다.", "detail": str(exc), "maps_url": maps_url, "places": []},
            status_code=502,
        )
