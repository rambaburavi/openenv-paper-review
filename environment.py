from pydantic import BaseModel
from tasks import TASKS
from graders import grade


class Observation(BaseModel):
    abstract: str
    stage: str


class Action(BaseModel):
    domain: str | None = None
    keywords: list[str] = []
    decision: str | None = None
    confidence: float | None = None


class Reward(BaseModel):
    score: float


class PaperReviewEnv:

    def __init__(self):

        self.tasks = TASKS
        self.index = 0
        self.current_task = None
        self.stage = None
        self.done = False
        self.partial_prediction = {}

    def reset(self):

        if self.index >= len(self.tasks):
            self.index = 0

        self.current_task = self.tasks[self.index]
        self.index += 1

        self.stage = "domain"
        self.done = False
        self.partial_prediction = {}

        return Observation(
            abstract=self.current_task["abstract"],
            stage=self.stage
        )

    def step(self, action: Action):

        reward_score = 0.0

        if self.stage == "domain":

            self.partial_prediction["domain"] = action.domain

            if action.domain == self.current_task["domain"]:
                reward_score = 0.4

            self.stage = "keywords"

            return Observation(
                abstract=self.current_task["abstract"],
                stage=self.stage
            ), Reward(score=reward_score), False, {}

        elif self.stage == "keywords":

            self.partial_prediction["keywords"] = action.keywords

            overlap = len(
                set(action.keywords) &
                set(self.current_task["keywords"])
            )

            reward_score = 0.3 * (
                overlap / max(len(self.current_task["keywords"]), 1)
            )

            self.stage = "decision"

            return Observation(
                abstract=self.current_task["abstract"],
                stage=self.stage
            ), Reward(score=reward_score), False, {}

        elif self.stage == "decision":

            self.partial_prediction["decision"] = action.decision
            self.partial_prediction["confidence"] = action.confidence

            total_score = grade(
                self.partial_prediction,
                self.current_task
            )

            self.done = True

            return Observation(
                abstract=self.current_task["abstract"],
                stage="completed"
            ), Reward(score=total_score), True, {}

    def state(self):

        return {
            "task": self.current_task,
            "stage": self.stage,
            "prediction": self.partial_prediction
        }