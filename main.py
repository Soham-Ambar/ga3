from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from q2_image_qa import router as q2_router
from q3_invoice_extract import router as q3_router
from q4_dynamic_extract import router as q4_router
from q6_audio_stats import router as q6_router
from q7_invoice_intelligence import router as q7_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Q2 - locked
app.include_router(q2_router)

# Q3 - locked
app.include_router(q3_router)

# Q4 - locked
app.include_router(q4_router)

# Q6 - temporarily skipped, but route remains available
app.include_router(q6_router)

# Q7 - Invoice Intelligence
app.include_router(q7_router)


@app.get("/")
def root():
    return {
        "message": "TDS GA3 API running",
        "endpoints": [
            "/answer-image",
            "/extract",
            "/dynamic-extract",
            "/audio-stats",
            "/invoice-intelligence",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }