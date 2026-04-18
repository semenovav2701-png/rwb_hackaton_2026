from fastapi import APIRouter, Depends
from src.app.schemas import Prediction
from src.app.application import Application
from src.app.depency_injection.get_application import get_application


router = APIRouter()

@router.post("/plan")
def plan(predictions: list[Prediction], app: Application = Depends(get_application)):
    return app.run(predictions)
    