import uvicorn

if __name__ == "__main__":
    print("Starting server...")
    print("View the family tree at: http://localhost:8000")
    print("To force refresh: http://localhost:8000/refresh")
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        workers=1,
        reload=True
    ) 