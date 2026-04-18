from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def get_status():
    return {"status": "ok"}
