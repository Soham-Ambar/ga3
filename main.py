print("MAIN: starting", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

print("MAIN: importing Q2", flush=True)
from q2_image_qa import router as q2_router
print("MAIN: Q2 imported", flush=True)

print("MAIN: importing Q3", flush=True)
from q3_invoice_extract import router as q3_router
print("MAIN: Q3 imported", flush=True)

print("MAIN: importing Q4", flush=True)
from q4_dynamic_extract import router as q4_router
print("MAIN: Q4 imported", flush=True)

print("MAIN: importing Q7", flush=True)
from q7_invoice_intelligence import router as q7_router
print("MAIN: Q7 imported", flush=True)

print("MAIN: importing Q8", flush=True)
from q8_semantic_search import router as q8_router
print("MAIN: Q8 imported", flush=True)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(q2_router)
app.include_router(q3_router)
app.include_router(q4_router)
app.include_router(q7_router)
app.include_router(q8_router)


@app.get("/")
def root():
    return {
        "message": "TDS GA3 API running",
        "endpoints": [
            "/answer-image",
            "/extract",
            "/dynamic-extract",
            "/invoice-intelligence",
            "/semantic-search",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


print("MAIN: FastAPI app ready", flush=True)