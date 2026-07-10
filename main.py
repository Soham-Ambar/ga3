from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from q2_image_qa import router as q2_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(q2_router)


@app.get("/")
def root():
    return {
        "message": "TDS GA3 API running",
        "endpoint": "/answer-image",
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }