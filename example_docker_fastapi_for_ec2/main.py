from datetime import datetime
from fastapi import FastAPI

app = FastAPI()


class TestAWS():
    def __init__(self):
        print('Example is create')
        self.time_now = datetime.now()

    def check_time(self):
        print(f"Time is now: {self.time_now}")
        return self.time_now

    def update_time(self):
        self.time_now = datetime.now()
        print('Time is update')
        print(f"Time is now: {self.time_now}")


ex_1 = TestAWS()


@app.get("/")
async def root():
    return {"message": "I'm health"}


@app.get("/get-time")
async def root():
    time_now = ex_1.check_time()
    return {"time": time_now}


@app.get("/upd-time")
async def root():
    ex_1.update_time()
    return {"status": "OK"}
