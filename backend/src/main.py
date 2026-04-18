from fastapi import FastAPI
from .app.endpoints.predictions import router as predictions_router
from .app.endpoints.plan import router as plan_router

server = FastAPI()
server.include_router(predictions_router)
server.include_router(plan_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="127.0.0.1", port=8000, reload=True)