from fastapi import APIRouter
from src.app.schemas import Prediction

router = APIRouter()

fake_database =[]

@router.get("/predictions")
def get_predictions():
    return fake_database


def add_prediction(pred: Prediction):
    new_pred = pred.model_dump()
    new_pred["id"] = len(fake_database) + 1

    fake_database.append(new_pred)
    return new_pred


@router.post("/predictions")
def create_multiply_predictions(preds: list[Prediction]):
    return [add_prediction(pred) for pred in preds]
    

# @app.put("/predictions/{task_id}")
# def update_prediction(task_id: int, task: Prediction):
#     for idx, t in enumerate(fake_database):
#         if t["id"] == task_id:
#             updated_task = task.dict()
#             updated_task["id"] = task_id
#             fake_database[idx] = updated_task
#             return updated_task

#     raise HTTPException(status_code=404, detail="Задача не найдена")


# @app.delete("/predictions/{task_id}")
# def delete_prediction(task_id: int):
#     for idx, t in enumerate(fake_database):
#         if t["id"] == task_id:
#             del fake_database[idx]
#             return {"message": "Задача успешно удалена"}

#     raise HTTPException(status_code=404, detail="Задача не найдена")