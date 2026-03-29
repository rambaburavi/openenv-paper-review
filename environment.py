from pydantic import BaseModel
from tasks import TASKS
from graders import grade


class Observation(BaseModel):
    abstract: str
    task_type: str


class Action(BaseModel):
    domain: str = None
    keywords: list[str] = []
    decision: str = None


class Reward(BaseModel):
    score: float


class PaperReviewEnv:

    def __init__(self):
        self.tasks = TASKS
        self.index = 0
        self.current_task = None
        self.done = False

    def reset(self):

        if self.index >= len(self.tasks):
            self.index = 0

        self.current_task = self.tasks[self.index]
        self.index += 1
        self.done = False

        return Observation(
            abstract=self.current_task["abstract"],
            task_type="paper_review"
        )

    def step(self, action: Action):

        score = grade(action.dict(), self.current_task)

        self.done = True

        observation = Observation(
            abstract=self.current_task["abstract"],
            task_type="completed"
        )

        return observation, Reward(score=score), self.done, {}

    def state(self):
        return self.current_task