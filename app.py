from fastapi import FastAPI
from environment import PaperReviewEnv, Action

app = FastAPI()

env = PaperReviewEnv()


@app.get("/")
def root():
    return {"message": "Paper Review OpenEnv is running"}


@app.get("/reset")
def reset():
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, info = env.step(action_obj)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()