from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from q2_image_qa import router as q2_router
from q3_invoice_extract import router as q3_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing Q2
app.include_router(q2_router)

# New Q3
app.include_router(q3_router)


@app.get("/")
def root():
    return {
        "message": "TDS GA3 API running",
        "endpoints": [
            "/answer-image",
            "/extract",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }