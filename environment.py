from pydantic import BaseModel
from tasks import TASKS
from graders import grade


class Observation(BaseModel):
    abstract: str
    task_type: str
    stage: str


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
        self.stage = 0
        self.partial_action = {}

    def reset(self):

        if self.index >= len(self.tasks):
            self.index = 0

        self.current_task = self.tasks[self.index]
        self.index += 1

        self.stage = 0
        self.done = False
        self.partial_action = {}

        return Observation(
            abstract=self.current_task["abstract"],
            task_type="paper_review",
            stage="domain"
        )

    def step(self, action: Action):

        reward_score = 0.0

        # STEP 1 → DOMAIN
        if self.stage == 0:
            self.partial_action["domain"] = action.domain

            if action.domain == self.current_task["domain"]:
                reward_score = 0.4

            self.stage = 1

            return Observation(
                abstract=self.current_task["abstract"],
                task_type="paper_review",
                stage="keywords"
            ), Reward(score=reward_score), False, {}

        # STEP 2 → KEYWORDS
        elif self.stage == 1:
            self.partial_action["keywords"] = action.keywords

            overlap = len(
                set(action.keywords) &
                set(self.current_task["keywords"])
            )

            reward_score = 0.3 * (
                overlap / max(len(self.current_task["keywords"]), 1)
            )

            self.stage = 2

            return Observation(
                abstract=self.current_task["abstract"],
                task_type="paper_review",
                stage="decision"
            ), Reward(score=reward_score), False, {}

        # STEP 3 → DECISION
        elif self.stage == 2:
            self.partial_action["decision"] = action.decision

            if action.decision == self.current_task["decision"]:
                reward_score = 0.3

            total_score = grade(
                self.partial_action,
                self.current_task
            )

            self.done = True

            return Observation(
                abstract=self.current_task["abstract"],
                task_type="completed",
                stage="finished"
            ), Reward(score=total_score), True, {}

    def state(self):

        return {
            "task": self.current_task,
            "stage": self.stage,
            "partial_action": self.partial_action
        }